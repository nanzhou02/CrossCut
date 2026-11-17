"""Path regression tests using real PNG IO, without importing the training framework.

Run with: python -m unittest discover -s tests -p test_isaid_paths.py -v
"""
import ast
import importlib.util
import os
from pathlib import Path
import pickle
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


REPO_ROOT = Path(os.environ.get('CROSSCUT_TEST_ROOT', Path(__file__).resolve().parents[1]))


class DatasetBase:
    def __init__(self, **kwargs):
        self.base_kwargs = kwargs


class Sample:
    def __init__(self, image, mask, objects_ids, sample_id):
        self.image = image
        self.mask = mask
        self.objects_ids = objects_ids
        self.sample_id = sample_id


def load_datasets():
    # Only framework containers are replaced; loader code and PNG IO run unchanged.
    base = ModuleType('isegm.data.base')
    base.ISDataset = DatasetBase
    sample = ModuleType('isegm.data.sample')
    sample.DSample = Sample
    spec = importlib.util.spec_from_file_location(
        'isaid_under_test', REPO_ROOT / 'isegm/data/datasets/isaid.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'isegm.data.base': base, 'isegm.data.sample': sample}):
        spec.loader.exec_module(module)
    return module


class ISAIDPathsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets = load_datasets()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='crosscut-paths-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.annotations = self.root / 'annotations'
        self.images = self.root / 'custom images'
        self.rgb = np.array([[[11, 22, 33], [44, 55, 66]]], dtype=np.uint8)
        self.mask = np.array([[[0, 0, 1], [0, 0, 0]]], dtype=np.uint8)
        for split in ('train', 'val', 'test'):
            image_dir = self.images / split / 'images'
            mask_dir = self.annotations / split / 'Semantic_masks/images'
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            Image.fromarray(self.rgb).save(image_dir / 'P0001.png')
            Image.fromarray(self.mask).save(mask_dir / 'P0001_instance_color_RGB.png')

    def assert_sample(self, dataset, split):
        self.assertEqual(dataset._images_path, self.images / split / 'images')
        self.assertEqual(dataset._sem_mask_path,
                         self.annotations / split / 'Semantic_masks/images')
        self.assertNotIn('dota_path', dataset.base_kwargs)
        sample = dataset.get_sample(0)
        np.testing.assert_array_equal(sample.image, self.rgb)
        np.testing.assert_array_equal(sample.mask, [[1, 0]])
        self.assertEqual(sample.objects_ids, [1])

    def test_custom_roots_and_all_supported_splits(self):
        for loader, splits in ((self.datasets.iSAIDDataset, ('train', 'val')),
                               (self.datasets.iSAIDEvaluationDataset, ('train', 'val', 'test'))):
            for split in splits:
                for root in (self.images, str(self.images)):
                    with self.subTest(loader=loader.__name__, split=split, root_type=type(root)):
                        dataset = loader(self.annotations, split, dota_path=root, epoch_len=7)
                        self.assert_sample(dataset, split)
                        self.assertEqual(dataset.base_kwargs, {'epoch_len': 7})

    def test_legacy_defaults_and_positional_split(self):
        workdir = self.root / 'work'
        workdir.mkdir()
        default_root = self.root / 'datasets/DOTA-1.0'
        default_root.parent.mkdir(parents=True)
        self.images.rename(default_root)
        previous = Path.cwd()
        os.chdir(workdir)
        self.addCleanup(os.chdir, previous)
        for loader, default_split in ((self.datasets.iSAIDDataset, 'train'),
                                      (self.datasets.iSAIDEvaluationDataset, 'val')):
            for split in ('train', 'val'):
                with self.subTest(loader=loader.__name__, split=split):
                    dataset = loader(self.annotations, split)
                    self.assertEqual(dataset._images_path, Path('../datasets/DOTA-1.0') / split / 'images')
                    np.testing.assert_array_equal(dataset.get_sample(0).image, self.rgb)
            self.assertEqual(loader(self.annotations).dataset_split, default_split)

    def test_relative_custom_root(self):
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        for loader in (self.datasets.iSAIDDataset, self.datasets.iSAIDEvaluationDataset):
            dataset = loader('annotations', 'val', dota_path='custom images')
            self.assertEqual(dataset._images_path, Path('custom images/val/images'))
            np.testing.assert_array_equal(dataset.get_sample(0).image, self.rgb)

    def test_evaluation_cache_stays_with_annotations_and_can_be_reused(self):
        loader = self.datasets.iSAIDEvaluationDataset
        first = loader(self.annotations, 'val', dota_path=self.images)
        cache = self.annotations / 'val_images_and_ids_list.pkl'
        self.assertTrue(cache.is_file())
        with cache.open('rb') as stream:
            self.assertEqual(pickle.load(stream), [('P0001_instance_color_RGB.png', 1)])
        original_cache = cache.read_bytes()
        second_images = self.root / 'relocated images'
        self.images.rename(second_images)
        second = loader(self.annotations, 'val', dota_path=second_images)
        self.assertEqual(second.dataset_samples, first.dataset_samples)
        np.testing.assert_array_equal(second.get_sample(0).image, self.rgb)
        self.assertEqual(cache.read_bytes(), original_cache)
        self.assertFalse(list(second_images.rglob('*.pkl')))

    def test_training_entrypoints_use_configured_image_root(self):
        cfg = SimpleNamespace(ISAID_PATH=self.annotations, DOTA_PATH=self.images)
        for name in ('plainvit_base448_crosscut.py', 'plainvit_large448_crosscut.py'):
            tree = ast.parse((REPO_ROOT / 'models/iter_mask' / name).read_text(encoding='utf-8'))
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == 'iSAIDDataset']
            self.assertEqual(len(calls), 2)
            splits = []
            for call in calls:
                context = dict(cfg=cfg, iSAIDDataset=self.datasets.iSAIDDataset,
                               train_augmentator=None, val_augmentator=None, points_sampler=None)
                dataset = eval(compile(ast.Expression(call), str(name), 'eval'), context)
                self.assert_sample(dataset, dataset.dataset_split)
                splits.append(dataset.dataset_split)
            self.assertCountEqual(splits, ['train', 'val'])

    def test_evaluation_entrypoint_uses_configured_image_root(self):
        path = REPO_ROOT / 'isegm/inference/utils.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'get_dataset')
        context = {'iSAIDEvaluationDataset': self.datasets.iSAIDEvaluationDataset}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), context)
        cfg = SimpleNamespace(ISAID_PATH=self.annotations, DOTA_PATH=str(self.images))
        self.assert_sample(context['get_dataset']('iSAID', cfg), 'val')


if __name__ == '__main__':
    unittest.main()
