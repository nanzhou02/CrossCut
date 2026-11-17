"""Real PyTorch integration checks using synthetic data and a small CrossCut."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isegm.data.datasets.deepglobe import DeepglobeEvaluationDataset
from isegm.data.datasets.isaid import iSAIDDataset, iSAIDEvaluationDataset
from isegm.data.points_sampler import MultiPointSampler
from isegm.inference import utils
from isegm.model.is_plainvit_crosscut_model import PlainVitCrosscutModel
from isegm.model.modeling.transformer_helper.cross_entropy_loss import CrossEntropyLoss
from isegm.utils.misc import save_checkpoint


class PyTorchRuntimeTest(unittest.TestCase):
    def test_real_isaid_loader_and_dataloader(self):
        self.assertTrue(issubclass(iSAIDDataset, Dataset))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / 'DOTA/train/images'
            masks = root / 'iSAID/train/Semantic_masks/images'
            images.mkdir(parents=True)
            masks.mkdir(parents=True)
            rgb = np.full((32, 32, 3), [20, 40, 60], dtype=np.uint8)
            mask = np.zeros((32, 32, 3), dtype=np.uint8)
            mask[8:24, 8:24, 2] = 1
            Image.fromarray(rgb).save(images / 'P0001.png')
            Image.fromarray(mask).save(masks / 'P0001_instance_color_RGB.png')
            dataset = iSAIDDataset(root / 'iSAID', 'train', dota_path=root / 'DOTA',
                                   points_sampler=MultiPointSampler(max_num_points=2))
            batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
            self.assertEqual(tuple(batch['images'].shape), (1, 3, 32, 32))
            self.assertEqual(batch['images'].dtype, torch.float32)
            self.assertEqual(tuple(batch['points'].shape), (1, 4, 3))
            self.assertGreater(batch['instances'].sum().item(), 0)
            evaluation = iSAIDEvaluationDataset(root / 'iSAID', 'train', dota_path=root / 'DOTA')
            np.testing.assert_array_equal(evaluation.get_sample(0).image, rgb)

    def test_deepglobe_string_paths_and_rectangular_patch_crop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'train').mkdir()
            rgb = np.full((4, 6, 3), 80, dtype=np.uint8)
            mask = np.zeros((4, 6, 3), dtype=np.uint8)
            mask[2:, 3:] = 255
            Image.fromarray(rgb).save(root / 'train/example_sat.jpg')
            Image.fromarray(mask).save(root / 'train/example_mask.png')
            for filename in ('train.txt', 'test.txt'):
                (root / filename).write_text('example_sat.jpg\n', encoding='utf-8')
            original_imread = cv2.imread

            def string_imread(path, flags):
                self.assertIsInstance(path, str)
                return original_imread(path, flags)

            for split in ('train', 'val'):
                with self.subTest(split=split):
                    dataset = DeepglobeEvaluationDataset(root, split, patch_size=(2, 3), whole_size=(6, 4))
                    with patch('cv2.imread', side_effect=string_imread), \
                         patch('random.random', return_value=0.9), patch('random.shuffle'):
                        sample = dataset.get_sample(0)
                    self.assertEqual(sample.image.shape, (4, 6, 3))
                    if split == 'train':
                        np.testing.assert_array_equal(sample.get_object_mask(0), np.ones((4, 6)))
                    else:
                        self.assertEqual(sample.get_object_mask(0).sum(), 6)

    def test_vectorized_click_pooling_matches_historical_behavior(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(device=device, dtype=dtype):
                    torch.manual_seed(7)
                    feature = torch.randn(3, 5, 4, 6, device=device, dtype=dtype)
                    mask = torch.randint(-1, 3, (3, 1, 4, 6), device=device)
                    mask[0] = 0
                    expected = []
                    for index in range(3):
                        pixels = feature[index, :, mask[index, 0] > 0]
                        expected.append(pixels.mean(1) if pixels.shape[1] else feature.new_zeros(5))
                    actual = PlainVitCrosscutModel.get_mean_feature(None, feature, mask)
                    torch.testing.assert_close(actual, torch.stack(expected))
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(actual.device.type, device)

    @staticmethod
    def small_model(device):
        return PlainVitCrosscutModel(
            use_disks=True, norm_radius=5, with_prev_mask=True, slice_number=2,
            backbone_params=dict(img_size=(448, 448), patch_size=(16, 16), in_chans=3,
                                 embed_dim=24, depth=8, num_heads=3, mlp_ratio=2, qkv_bias=True),
            neck_params=dict(in_dim=24, out_dims=[8, 16, 24, 32]),
            head_params=dict(in_channels=[8, 16, 24, 32], in_index=[0, 1, 2, 3],
                             channels=16, num_classes=1, dropout_ratio=0, loss_decode=CrossEntropyLoss(),
                             align_corners=False, upsample='x1'),
            random_split=False).to(device)

    def test_forward_backward_and_checkpoint_round_trip(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(7)
        model = self.small_model(device)
        self.assertIsInstance(model, torch.nn.Module)
        image = torch.rand(1, 4, 32, 48, device=device)
        image[:, 3] = 0
        points = torch.tensor([[[8, 12, 0], [-1, -1, -1],
                                [24, 36, 1], [-1, -1, -1]]], device=device)
        original_points = points.clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        output = model(image, points)['instances']
        self.assertEqual(tuple(output.shape), (1, 1, 32, 48))
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(points, original_points)
        loss = output.square().mean()
        loss.backward()
        self.assertIsNotNone(model.mlp_pos[0].weight.grad)
        self.assertTrue(torch.isfinite(model.mlp_pos[0].weight.grad).all())
        optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = model(image, points)['instances']
        with tempfile.TemporaryDirectory() as temp:
            save_checkpoint(model, Path(temp), verbose=False)
            checkpoint = next(Path(temp).glob('*.pth'))
            restored = utils.load_is_model(checkpoint, device, eval_ritm=False)
            with torch.no_grad():
                actual = restored(image, points)['instances']
            torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
