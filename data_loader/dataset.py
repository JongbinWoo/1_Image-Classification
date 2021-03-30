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


#%%
class MaskDataset(Dataset):
    def __init__(self, data_root, transform=None):
        super(MaskDataset, self).__init__()
        self.info_df = pd.read_csv('/opt/ml/input/data/train/train.csv')
        self.folder_list = self._get_folder_list()
        self.len = len(self.folder_list) * 7
        self.data_root = data_root

        self.transform = True
               

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
        
        gender, age = self._get_class(folder_idx)
        
        img_path = os.path.join(folder_path, img_name)
        
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(img_path)
        # img = img/255.
        if self.transform:
            # img = self.transform(img)
            img = transforms.CenterCrop(224)(img)
            img = transforms.ToTensor()(img)
        
        
        return img, [gender, age, mask]

    def __len__(self):
        return self.len
    
    def _get_folder_list(self):
        return self.info_df.path.values

    def _get_class(self, idx):
        idx_data =  self.info_df.iloc[idx]
        return idx_data.gender, idx_data.age_class
     
#%%
def get_augmentation(size=224, use_flip=True, use_color_jitter=False, use_gray_scale=False, use_normalize=False):
    resize_crop = transforms.RandomResizedCrop(size=size)
    random_flip = transforms.RandomHorizontalFlip(p=0.5)
    color_jitter = transforms.RandomApply([
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    ], p=0.8)
    
    gray_scale = transforms.RandomGrayscale(p=0.2)
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    to_tensor = transforms.ToTensor()
    
    transforms_array = np.array([resize_crop, random_flip, color_jitter, gray_scale, to_tensor, normalize])
    transforms_mask = np.array([True, use_flip, use_color_jitter, use_gray_scale, True, use_normalize])
    
    # transform = transforms.Compose(transforms_array[transforms_mask])
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        to_tensor
    ])
    
    return transform

#%%
