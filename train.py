#%%
import torch
import argparse

#data
from data_loader.data_loader import get_loader
from data_loader.dataset import get_mnist, get_augmentation

#config
from config import get_config

#model
from model.model import MLP
from model import loss
from model.optimizer import get_optimizer 

#trianer
from trainer.trainer import Trainer


SEED = 42
torch.manual_seed(SEED)

# %%
def main(config):
    transform = get_augmentation(**config.TRAIN.AUGMENTATION)
    train_dataset, test_dataset = get_mnist(config.DATASET.ROOT, transform=transform)

    train_loader = get_loader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True)
    test_loader = get_loader(test_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True)
    
    mlp = MLP(input_features=784, hidden_size=256, output_features=config.DATASET.NUM_CLASSES)
    print('[Model Info]\n\n', mlp)
    optimizer = get_optimizer(optimizer_name = config.MODEL.OPTIM, 
                                        lr=config.TRAIN.BASE_LR, 
                                        model=mlp)
    import torch.nn as nn        ##
    loss = nn.CrossEntropyLoss() ##
    
    trainer = Trainer(mlp, optimizer, loss,  config, train_loader, test_loader)
    trainer.train()
    
# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLP')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    args = parser.parse_args()

    config = get_config(args)

    main(config)