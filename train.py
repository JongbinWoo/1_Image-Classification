#%%
import torch
import argparse

#data
from data_loader.data_loader import get_loader
from data_loader.dataset import MaskDataset, get_augmentation

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
    transform = get_augmentation(**config.TRAIN.AUGMENTATION)
    
    dataset = MaskDataset(config.PATH.ROOT, transform=transform)
    
    len_valid_set = int(config.DATASET.RATIO * len(dataset))
    len_train_set = len(dataset) - len_valid_set
    
    # class의 분포를 고려해서 samplig하는 방법: https://github.com/catalyst-team/catalyst
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [len_train_set, len_valid_set])

    train_loader = get_loader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE,
                              num_workers=config.TRAIN.NUM_WORKERS, shuffle=True)
    valid_loader = get_loader(valid_dataset, batch_size=config.TRAIN.BATCH_SIZE,
                              num_workers=config.TRAIN.NUM_WORKERS, shuffle=True)

    # MODEL
    model = VGG(config.DATASET.NUM_CLASSES)
    print('[Model Info]\n\n', model)
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
