import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from tqdm import tqdm


class InriaEvaluationDataset(ISDataset):
    def __init__(self, dataset_path, split='val', **kwargs):
        super(InriaEvaluationDataset, self).__init__(**kwargs)
        assert split in {'train', 'val', 'test'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / 'train' / 'images'
        self._mask_path = self.dataset_path / 'train' / 'gt'
        
        self.dataset_samples=[x.name for x in sorted(self._images_path.glob('*.tif*'))]
        
        if split in ['train']:
            with open(str(self.dataset_path / 'train.txt'), 'r') as f:
                self.dataset_samples = [line.strip() for line in f.readlines()]
        else:
            with open(str(self.dataset_path / 'test.txt'), 'r') as f:
                self.dataset_samples = [line.strip() for line in f.readlines()]
        
        self.dataset_samples=self.get_images_and_ids_list()
        
    def get_sample(self, index) -> DSample:

        image_name, slices_id = self.dataset_samples[index]
        mask_name = image_name
        
        image_path = str(self._images_path / image_name)
        mask_path = str(self._mask_path / mask_name)

        image=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        instances_mask = (np.array(Image.open(mask_path))).astype(np.int32)
        mask = (instances_mask>127).astype(np.int32)
        n = slices_id
        image = image[n[0]*1000:(n[0]+1)*1000,n[1]*1000:(n[1]+1)*1000,:] 
        instances_mask = mask[n[0]*1000:(n[0]+1)*1000,n[1]*1000:(n[1]+1)*1000]

        return DSample(image, instances_mask, objects_ids=[1], sample_id=index)
    
    def get_images_and_ids_list(self):
        pkl_path = self.dataset_path / f'{self.dataset_split}_images_and_ids_list.pkl'

        if pkl_path.exists():
            with open(str(pkl_path), 'rb') as fp:
                images_and_ids_list = pkl.load(fp)
        else:
            images_and_ids_list = []

            for sample in tqdm(self.dataset_samples):
                inst_info_path = str(self._mask_path / sample)
                instances_mask = (np.array(Image.open(inst_info_path))).astype(np.int32)
                instances_mask = (instances_mask>127).astype(np.int32)
                slices=[(i,j) for i in range(5) for j in range(5)]
                slice_masks=[instances_mask[n[0]*1000:(n[0]+1)*1000,n[1]*1000:(n[1]+1)*1000] for n in slices]
                
                slices_ids=[]
                for slices_id, slice_mask in zip(slices,slice_masks):
                    if not np.all(slice_mask==0):
                        slices_ids.append(slices_id)
    
                for slices_id in slices_ids:
                    images_and_ids_list.append((sample, slices_id))
            
            with open(str(pkl_path), 'wb') as fp:
                pkl.dump(images_and_ids_list, fp)

        return images_and_ids_list
    

