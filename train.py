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


SEED = 42
torch.manual_seed(SEED)

# %%
def main(config):
    # DATA SETTING
    # transform = get_augmentation(**config.TRAIN.AUGMENTATION)
    
    # dataset = MaskDataset(config.PATH.ROOT, transform=transform)
    
    # len_valid_set = int(config.DATASET.RATIO * len(dataset))
    # len_train_set = len(dataset) - len_valid_set
    
    # # class의 분포를 고려해서 samplig하는 방법: https://github.com/catalyst-team/catalyst
    # # train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [len_train_set, len_valid_set],
    # #                                                              generator=iter(ImbalancedDatasetSampler(dataset.info_df)))
    # idx_sampler = iter(ImbalancedDatasetSampler(dataset.info_df))
    # valid_indices = islice(idx_sampler, len_valid_set)

    # train_loader = get_loader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE,
    #                           num_workers=config.TRAIN.NUM_WORKERS, shuffle=True)
    # valid_loader = get_loader(valid_dataset, batch_size=config.TRAIN.BATCH_SIZE,
    #                           num_workers=config.TRAIN.NUM_WORKERS, shuffle=True)

    transform = get_augmentation(**config.TRAIN.AUGMENTATION)
    
    dataset = MaskDataset(config.PATH.ROOT, transform=transform)

    len_valid_set = int(config.DATASET.RATIO * len(dataset.info_df))
    len_train_set = len(dataset) - len_valid_set

    idx_sampler = iter(ImbalancedDatasetSampler(dataset.info_df)) 
    valid_indices = set(islice(idx_sampler, len_valid_set))
    train_indices = set(range(len(dataset.info_df))) - valid_indices

    def return_image_indices(i):
        return i*7, i*7+1, i*7+2, i*7+3, i*7+4, i*7+5, i*7+6

    train_indices = [x for i in train_indices for x in return_image_indices(i)]
    valid_indices = [x for i in valid_indices for x in return_image_indices(i)]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)


    train_loader = get_loader(dataset, batch_size=config.TRAIN.BATCH_SIZE,
                                num_workers=config.TRAIN.NUM_WORKERS, sampler=train_sampler)
    valid_loader = get_loader(dataset, batch_size=config.TRAIN.BATCH_SIZE,
                                num_workers=config.TRAIN.NUM_WORKERS, sampler=valid_sampler)
    # MODEL
    model = VGG(config.DATASET.NUM_CLASSES)
    # print('[Model Info]\n\n', model)
    optimizer = get_optimizer(optimizer_name = config.MODEL.OPTIM, 
                              lr=config.TRAIN.BASE_LR, 
                              model=model)
    import torch.optim as optim
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    import torch.nn as nn        ##
    loss = nn.CrossEntropyLoss() ##
    
    trainer = Trainer(model, optimizer, scheduler, loss,  config, train_loader, valid_loader)
    trainer.train()
    
# %%
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='MLP')
    # parser.add_argument('--epochs', default=10, type=int)
    # parser.add_argument('--batch_size', default=256, type=int)
    # parser.add_argument('--lr', default=0.001, type=float)

    # args = parser.parse_args()

    config = get_config()

    main(config)
# %%

