import pickle as pkl
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from tqdm import tqdm


class iSAIDDataset(ISDataset):
    def __init__(self, dataset_path, split='train', **kwargs):
        super(iSAIDDataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = Path('../datasets/DOTA-1.0') / split / 'images'
        self._sem_mask_path = self.dataset_path / split / 'Semantic_masks' / 'images'
        self.dataset_samples = [x.name for x in sorted(self._sem_mask_path.glob('*.png*'))]
        

    def get_sample(self, index) -> DSample:

        mask_name = self.dataset_samples[index]
        image_name = mask_name.split('_')[0]+ ".png"
        
        image_path = str(self._images_path / image_name)
        mask_path = str(self._sem_mask_path / mask_name)

        
        image=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        instances_mask = (np.array(Image.open(mask_path))).astype(np.int32)
        blue_channel, green_channel, red_channel = cv2.split(instances_mask)
        mask = red_channel + 100 * green_channel + 10000 * blue_channel
        mask = mask.astype(np.int32)
        instances_ids=[x for x in np.unique(mask) if x != 0]
        
        
        
        return DSample(image, mask, objects_ids=instances_ids, sample_id=index)
    

class iSAIDEvaluationDataset(ISDataset):
    def __init__(self, dataset_path, split='val', **kwargs):
        super(iSAIDEvaluationDataset, self).__init__(**kwargs)
        assert split in {'train', 'val', 'test'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = Path('../datasets/DOTA-1.0') / split / 'images'
        self._sem_mask_path = self.dataset_path / split / 'Semantic_masks' / 'images'
        self.dataset_samples = [x.name for x in sorted(self._sem_mask_path.glob('*.png*'))]
        self.dataset_samples = self.get_images_and_ids_list()
        

    def get_sample(self, index) -> DSample:

        mask_name, instance_id = self.dataset_samples[index]
        image_name = mask_name.split('_')[0]+ ".png"
        
        image_path = str(self._images_path / image_name)
        mask_path = str(self._sem_mask_path / mask_name)

        
        image=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        instances_mask = (np.array(Image.open(mask_path))).astype(np.int32)
        blue_channel, green_channel, red_channel = cv2.split(instances_mask)
        instances_mask = red_channel + 100 * green_channel + 10000 * blue_channel
        instances_mask = instances_mask.astype(np.int32)
        instances_mask[instances_mask != instance_id] = 0
        instances_mask[instances_mask > 0] = 1


        return DSample(image, instances_mask, objects_ids=[1], sample_id=index)
    
    def get_images_and_ids_list(self):
        pkl_path = self.dataset_path / f'{self.dataset_split}_images_and_ids_list.pkl'

        if pkl_path.exists():
            with open(str(pkl_path), 'rb') as fp:
                images_and_ids_list = pkl.load(fp)
        else:
            images_and_ids_list = []

            for sample in tqdm(self.dataset_samples):
                inst_info_path = str(self._sem_mask_path / sample)
                instances_mask = (np.array(Image.open(inst_info_path))).astype(np.int32)
                blue_channel, green_channel, red_channel = cv2.split(instances_mask)
                instances_mask = red_channel + 100 * green_channel + 10000 * blue_channel
                instances_mask = instances_mask.astype(np.int32)
                instances_ids = [x for x in np.unique(instances_mask) if x != 0]

                for instances_id in instances_ids:
                    images_and_ids_list.append((sample, instances_id))

            with open(str(pkl_path), 'wb') as fp:
                pkl.dump(images_and_ids_list, fp)

        return images_and_ids_list
    