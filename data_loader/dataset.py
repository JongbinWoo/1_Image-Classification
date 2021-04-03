#%%
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset 
from torchvision import transforms
import pandas as pd
import cv2
import os 
from glob import glob 
import numpy as np
from torchvision.transforms.transforms import CenterCrop
import albumentations as A
from albumentations.pytorch import ToTensorV2
#%%
class MaskDataset(Dataset):
    def __init__(self, data_root, transform=None):
        super(MaskDataset, self).__init__()
        self.info_df = pd.read_csv('/opt/ml/input/data/train/train.csv')
        self.folder_list = self._get_folder_list()
        self.len = len(self.folder_list) * 7
        self.data_root = data_root

        self.transform = transform
               

    def __getitem__(self, index):
        folder_idx, img_idx = divmod(index, 7)
        folder_path = self.folder_list[folder_idx]
        folder_path = os.path.join(self.data_root, folder_path)
        
        img_list = glob(os.path.join(folder_path, '*.jpg'))
        img_list = list(map(os.path.basename, img_list))
        img_name = img_list[img_idx]
        
        if 'normal' in img_name: 
            mask = 2
        elif 'incorrect_mask' in img_name: 
            mask = 1
        else:
            mask = 0
        
        gender_age_class, age = self._get_class(folder_idx)
        
        img_path = os.path.join(folder_path, img_name)
        
        img = cv2.imread(img_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image = Image.open(image_path)
        # image = image/255.
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, [gender_age_class, mask, age]

    def __len__(self):
        return self.len
    
    def _get_folder_list(self):
        return self.info_df.path.values

    def _get_class(self, idx):
        idx_data =  self.info_df.iloc[idx]
        return idx_data.gender_age_class, idx_data.age
     
#%%
def get_augmentation():
    # resize_crop = transforms.RandomResizedCrop(size=size)
    # random_flip = transforms.RandomHorizontalFlip(p=0.5)
    # color_jitter = transforms.RandomApply([
    #     transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    # ], p=0.8)
    
    # gray_scale = transforms.RandomGrayscale(p=0.2)
    normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = ToTensorV2()
    
    # transforms_array = np.array([resize_crop, random_flip, color_jitter, gray_scale, to_tensor, normalize])
    # transforms_mask = np.array([True, use_flip, use_color_jitter, use_gray_scale, True, use_normalize])
    
    # transform = transforms.Compose(transforms_array[transforms_mask])
    transform = A.Compose([
        A.CenterCrop(p=1, height=224, width=224),
        # A.HorizontalFlip(p=0.5),
        # A.OpticalDistortion(p=0.5),
        # A.GridDropout(p=0.5),
        # A.ShiftScaleRotate(p=0.5),
        # A.ElasticTransform(p=0.5),
        normalize,       
        to_tensor
    ])
    
    return transform


#%%
import torch
import torch.utils.data
from collections import defaultdict

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = (list(range(len(dataset)))) 

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.dataset = dataset
        self.num_samples = len(self.dataset) 

        # distribution of classes in the dataset 
        self.label_to_count = defaultdict(int)
        for idx in self.indices:
            label = self._get_label(idx)
            self.label_to_count[label] += 1
           
                
        # weight for each sample
        weights = [1.0 / self.label_to_count[self._get_label(idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, idx):
        return self.dataset.iloc[idx].gender_age_class
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=False))

    def __len__(self):
        return self.num_samples
#%%
# %%
