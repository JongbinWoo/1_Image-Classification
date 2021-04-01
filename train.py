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
from model.model import DenseNet, EfficientNet_b0
from model import loss
from model.optimizer import get_optimizer 

#trianer
from trainer.trainer import Trainer

import wandb

SEED = 42
torch.manual_seed(SEED)

# %%
def main(config): # wandb_config):
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

    idx_sampler = ImbalancedDatasetSampler(dataset.info_df)
    class_to_count = idx_sampler.label_to_count
    class_weight = torch.tensor([class_to_count[i]/len(dataset.info_df) for i in range(6)])
    idx_sampler = iter(idx_sampler) 
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
    gender_age_model = DenseNet(6, config.MODEL.HIDDEN, config.MODEL.FREEZE)
    mask_model = EfficientNet_b0(3, config.MODEL.HIDDEN, config.MODEL.FREEZE)

    # print('[Model Info]\n\n', model)
    gender_age_optimizer = get_optimizer(optimizer_name = config.MODEL.OPTIM, 
                              lr=config.TRAIN.BASE_LR, 
                              model=gender_age_model)
    mask_optimizer = get_optimizer(optimizer_name = config.MODEL.OPTIM, 
                              lr=config.TRAIN.BASE_LR, 
                              model=mask_model)

    import torch.optim as optim
    gender_age_scheduler = optim.lr_scheduler.ReduceLROnPlateau(gender_age_optimizer, 'min', patience=2)
    mask_scheduler = optim.lr_scheduler.ReduceLROnPlateau(mask_optimizer, 'min', patience=2)

    import torch.nn as nn        ##
    # print(class_weight)
    gender_age_loss = nn.CrossEntropyLoss(weight=class_weight.cuda()) ##
    mask_loss = nn.CrossEntropyLoss(weight=torch.tensor([5/7, 1/7, 1/7]).cuda()) ##
    
    gender_age_model_set = {
        'model': gender_age_model,
        'optimizer': gender_age_optimizer,
        'scheduler': gender_age_scheduler,
        'criterion': gender_age_loss
    }
    mask_model_set = {
        'model': mask_model,
        'optimizer': mask_optimizer,
        'scheduler': mask_scheduler,
        'criterion': mask_loss
    }
    trainer = Trainer(gender_age_model_set, mask_model_set,  config, train_loader, valid_loader)
    trainer.train(config.TRAIN.EPOCH)#wandb_config['epochs'])
    
# %%
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='MLP')
    # parser.add_argument('--epochs', default=10, type=int)
    # parser.add_argument('--batch_size', default=256, type=int)
    # parser.add_argument('--lr', default=0.001, type=float)

    # args = parser.parse_args()
    config = get_config()
    
    # ######  WANDB ##########
    # sweep_config = {
    #     'method': 'random',
    #     'metric': {
    #         'name': 'val_acc',
    #         'goal': 'maximize'
    #     },
    #     'parameters': {
    #         'epochs': {
    #             'values': [2, 4, 6, 8]
    #         },
    #         'lr': {
    #             'distribution': 'uniform',
    #             'min': 0.00001,
    #             'max': 0.0001
    #         },
    #         'hidden_dim': {
    #             'values': [128, 256, 512]
    #         },
    #     }
    # }
    
    # config_default = {
    #     'lr' : config.TRAIN.BASE_LR,
    #     'hidden_dim': config.MODEL.HIDDEN,
    #     'epochs': config.TRAIN.EPOCH
    # }
    # wandb.init(config=config_default, project='Stage-1')
    # wandb_config = wandb.config

    # sweep_id = wandb.sweep(sweep_config, project='Stage-1')
    
    # wandb.agent(sweep_id, main(config, wandb_config))
    
    # print('FINISHED')

    main(config)

