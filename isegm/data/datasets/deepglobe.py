import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
import random
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class DeepglobeEvaluationDataset(ISDataset):
    def __init__(self, dataset_path, split='train', patch_size=448, whole_size=(896, 896), **kwargs):
        super(DeepglobeEvaluationDataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / 'train'
        self.patch_size = patch_size
        self.whole_size = whole_size

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.jpg*'))]

        if self.dataset_split == 'val':
            with open(str(self.dataset_path / 'test.txt'), 'r') as f:
                self.dataset_samples = [line.strip() for line in f.readlines()]
        elif self.dataset_split == 'train':
            with open(str(self.dataset_path / 'train.txt'), 'r') as f:
                self.dataset_samples = [line.strip() for line in f.readlines()]

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        mask_name = image_name.replace('sat.jpg', 'mask.png')

        image_path = self._images_path / image_name
        mask_path = self._images_path / mask_name

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 127).astype(np.uint8)

        p = random.random()
        if self.dataset_split == 'train':
            if p > 0.8:
                image = cv2.resize(image, self.whole_size, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, self.whole_size, interpolation=cv2.INTER_NEAREST)

                slices = [(0, 0), (1, 0), (0, 1), (1, 1)]
                random.shuffle(slices)
                slice_images = [image[n[0] * self.patch_size:(n[0] + 1) * self.patch_size,
                                n[1] * self.patch_size:(n[1] + 1) * self.patch_size, :] for n in slices]
                slice_masks = [mask[n[0] * self.patch_size:(n[0] + 1) * self.patch_size,
                               n[1] * self.patch_size:(n[1] + 1) * self.patch_size] for n in slices]
                for slice_image, slice_mask in zip(slice_images, slice_masks):
                    if not np.all(slice_mask == 0):
                        mask = cv2.resize(slice_mask, self.whole_size, interpolation=cv2.INTER_NEAREST)
                        image = cv2.resize(slice_image, self.whole_size, interpolation=cv2.INTER_LINEAR)
                        break

        return DSample(image, mask, objects_ids=[1], sample_id=index)
