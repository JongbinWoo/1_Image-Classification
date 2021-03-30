#%%
import torch
import argparse
from itertools import islice
from torch.utils.data.sampler import SubsetRandomSampler
#data
from data_loader.data_loader import get_loader
from data_loader.dataset import ImbalancedDatasetSampler, MaskDataset, get_augmentation

#config
from config import get_config

#model
from model.model import VGG
from model import loss
from model.optimizer import get_optimizer 

#trianer
from trainer.trainer import Trainer


config = get_config()
transform = get_augmentation(**config.TRAIN.AUGMENTATION)
    
dataset = MaskDataset(config.PATH.ROOT, transform=transform)

len_valid_set = int(config.DATASET.RATIO * len(dataset.info_df))
len_train_set = len(dataset) - len_valid_set

idx_sampler = iter(ImbalancedDatasetSampler(dataset.info_df)) 
valid_indices = set(islice(idx_sampler, len_valid_set))
train_indices = set(range(len(dataset.info_df))) - valid_indices

def return_image_indices(i):
    return i, i*2, i*3, i*4, i*5, i*6, i*7

train_indices = [x for i in train_indices for x in return_image_indices(i)]
valid_indices = [x for i in valid_indices for x in return_image_indices(i)]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)


train_loader = get_loader(dataset, batch_size=config.TRAIN.BATCH_SIZE,
                              num_workers=config.TRAIN.NUM_WORKERS, sampler=train_sampler)
valid_loader = get_loader(dataset, batch_size=config.TRAIN.BATCH_SIZE,
                              num_workers=config.TRAIN.NUM_WORKERS, sampler=valid_sampler)