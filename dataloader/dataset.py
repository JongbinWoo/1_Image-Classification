#%%
import os
from albumentations.augmentations.functional import iso_noise
from albumentations.augmentations.transforms import ISONoise
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
# %%
#%%
class MaskDataset(Dataset):
    def __init__(self,
                 df, 
                 root_dir='/opt/ml/input/data/train/image',
                 transform=ToTensorV2):
        super(MaskDataset, self).__init__()
        self.root_dir = root_dir
        self.info_df = df
        self.transform = transform
               

    def __getitem__(self, index):
        folder_idx, img_idx = divmod(index, 7)
        folder_path = self.info_df.loc[folder_idx].path
        folder_path = os.path.join(self.root_dir, folder_path)
        
        # img_path = os.path.join(folder_path, 'normal.jpg')

        img_list = glob(os.path.join(folder_path, '*.jpg'))
        img_path = img_list[img_idx]
        
        img = cv2.imread(img_path) # numpy ndarray type으로 리턴
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image = Image.open(image_path)
        # image = image/255.
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        gender_age_class, age = self._get_class(folder_idx)
        
        return image, [gender_age_class, age]

    def __len__(self):
        return len(self.info_df) * 7

    def _get_class(self, idx):
        idx_data =  self.info_df.loc[idx]
        return idx_data.gender_age_class, idx_data.age
     


# %%
def get_augmentation(config):
    normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = ToTensorV2()
    transforms = {}
    train_transform = A.Compose([
        A.CenterCrop(p=1, height=config.TRAIN.HEIGHT, width=config.TRAIN.WIDTH),
        A.Cutout(num_holes=4, max_h_size=10, max_w_size=10, p=0.3),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                 val_shift_limit=0.2, p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.2, 
                                       contrast_limit=0.2, p=0.9),
        ],p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ISONoise(p=0.3),
        normalize,       
        to_tensor
    ])
    transforms['train'] = train_transform
    valid_transform = A.Compose([
        A.CenterCrop(p=1, height=config.TRAIN.HEIGHT, width=config.TRAIN.WIDTH),
        normalize,       
        to_tensor
    ])
    transforms['valid'] = valid_transform

    return transforms


#%%
def generate_cutmix_image(image_batch, image_batch_labels):
    """ Generate a CutMix augmented image from a batch 
    Args:
        - image_batch: a batch of input images
        - image_batch_labels: labels corresponding to the image batch
        - beta: a parameter of Beta distribution.
    Returns:
        - CutMix image batch, updated labels
    """
    # generate mixed sample
    lam = 0.5 #np.random.beta(beta, beta)
    
    rand_index = np.random.permutation(len(image_batch))
    target_a = image_batch_labels
    target_b = image_batch_labels[rand_index]
    # bbx1, bby1, bbx2, bby2 = rand_bbox(image_batch[0].shape, lam)
    # image_batch_updated = image_batch.copy()
    image_batch_updated = image_batch.clone().detach()
    # image_batch_updated = torch.tensor(image_batch)
    image_batch_updated[:, :, :, :128] = image_batch[rand_index, :, :, :128]
    
    # adjust lambda to exactly match pixel ratio
    # lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image_batch.shape[1] * image_batch.shape[2]))
    label = target_a * lam + target_b * (1. - lam)
    
    return image_batch_updated, label

#%%
def balance_data(class_size,df):
    train_df = df.groupby(['gender_age_class']).apply(lambda x: x.sample(class_size, replace = True)).reset_index(drop = True)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    # print('New Data Size:', train_df.shape[0], 'Old Size:', df.shape[0])
    # print(train_df['gender_age_class'].value_counts())#hist(figsize = (10, 5))
    return train_df


# %%

# %%
class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        # image = Image.open(self.img_paths[index])
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image

    def __len__(self):
        return len(self.img_paths)

def get_test_transforms(CFG):
    return A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])