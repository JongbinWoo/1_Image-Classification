#%%
import pandas as pd
#data
from dataloader.dataset import MaskDataset, balance_data , get_augmentation
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

#config
from config import get_config

#model
from model.model import DenseNet, EfficientNet_b0, ResNext
from model.loss import Cumbo_Loss, CustomLoss, F1Loss, TaylorCrossEntropyLoss, loss_kd_regularization
from model.optimizer import get_optimizer, get_adamp
from madgrad import MADGRAD
import torch.optim as optim

#trianer
from trainer.trainer import Trainer

from utils import seed_everything



# %%
def main(config): 
    
    transforms = get_augmentation(config)
    
    # gender_age_class 의 분포를 고려해서 fold를 나눠준다
    df = pd.read_csv('/opt/ml/input/data/train/train.csv')
    skf = StratifiedKFold(n_splits=config.TRAIN.KFOLD, shuffle=True, random_state=42)
    for fold, (_, val_) in enumerate(skf.split(X=df, y=df.gender_age_class)):
        df.loc[val_, 'kfold'] = int(fold)
    df['kfold'] = df['kfold'].astype(int)

    # 각 fold에 대해 dataset&dataloader를 만들고 list에 저장해둔다.
    dataloaders = []
    for fold in range(config.TRAIN.KFOLD):
        train_df = df[df.kfold != fold]
        valid_df = df[df.kfold == fold]

        # Oversampling
        # https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-with-resnet50-oversampling/notebook
        train_df = balance_data(train_df.pivot_table(index='gender_age_class', aggfunc=len).max().max(),train_df)
        valid_df = valid_df.reset_index(drop=True)
        train_dataset = MaskDataset(train_df, config.PATH.ROOT, transforms['train'])
        valid_dataset = MaskDataset(valid_df, config.PATH.ROOT, transforms['valid'])

        train_loader = DataLoader(train_dataset,
                                  batch_size=config.TRAIN.BATCH_SIZE,
                                  num_workers=config.TRAIN.NUM_WORKERS,
                                  pin_memory=True,
                                  shuffle=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=config.TRAIN.BATCH_SIZE,
                                  num_workers=config.TRAIN.NUM_WORKERS,
                                  pin_memory=True,
                                  shuffle=False)
        dataloaders.append((train_loader, valid_loader))

    # loss function 선택
    loss_collection = {
        'TaylorCE': TaylorCrossEntropyLoss(),
        'CE': CustomLoss(config.TRAIN.T),
        'F1Loss': F1Loss(),
        'KD-Reg': loss_kd_regularization(config),
        'Cumbo': Cumbo_Loss(config)
    }
    loss = loss_collection[config.TRAIN.LOSS]
    print(f'Loss : {config.TRAIN.LOSS}')

    f1_scores = []
    for fold, dataloader in enumerate(dataloaders):
        print(f'\n----------- FOLD {fold} TRAINING START --------------\n')
        model = EfficientNet_b0(6, True, config.MODEL.FREEZE)
        # model = DenseNet(6, config.MODEL.HIDDEN, config.MODEL.FREEZE)
        # model = ResNext(6, config.MODEL.FREEZE)
        optimizer = MADGRAD(model.parameters(), lr=1e-4)
        # optimizer = get_adamp(lr=config.TRAIN.BASE_LR, model=model, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0= 10, T_mult= 1, eta_min= 1e-6, last_epoch=-1)
        model_set = {
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'criterion': loss
        }
        train_loader = dataloader[0]
        valid_loader = dataloader[1]
        trainer = Trainer(model_set, config, train_loader, valid_loader, fold)
        best_f1 = trainer.train(config.TRAIN.EPOCH)
        print(f'\nFOLD F{fold}: {best_f1:.3f}\n')
        f1_scores.append(best_f1)
    
    print(f'MEAN F1 - {sum(f1_scores)/len(f1_scores)}')
# %%
if __name__ == '__main__':
    config = get_config()
    seed_everything(42)
    
    main(config)

