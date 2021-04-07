#%%
import os
import shutil
from pathlib import Path
from collections import defaultdict, Counter
from utils import seed_everything
from model.model import EfficientNet_b0
from dataloader.dataset import TestDataset
from dataloader.dataset import get_test_transforms
import scipy as sp
import numpy as np
import pandas as pd


from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

import warnings 
warnings.filterwarnings('ignore')

# ====================================================
# Helper functions
# ====================================================
def load_state(model_path):
    model = EfficientNet_b0(6, pretrained=False)
    try:  # single GPU model_file  
        state_dict = torch.load(model_path)['model']
        model.load_state_dict(state_dict, strict=True)
    except:  # multi GPU model_file
        state_dict = torch.load(model_path)['model']
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}

    return state_dict


def inference(model, states, test_loader, device):
    model.to(device)
    probs = []
    for i, (images) in enumerate(test_loader):
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
            avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
            print(f'avg_preds: {avg_preds}')
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
        print(f'probs: {probs}')
    probs = np.concatenate(probs)
    return probs

class CFG:
    debug=False
    num_workers=8
    model_name='resnext50_32x4d'
    size=512
    batch_size=32
    seed=2020
    target_size=5
    target_col='label'
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    inference=True
    OUTPUT_DIR = './'
    MODEL_DIR = '/content/drive/MyDrive/kaggle/inference/cassava-resnext50-32x4d-weights'
    TEST_PATH = './test_images'

def main(config):

    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed_everything(seed=config.seed)

    test = pd.read_csv('./sample_submission.csv')
    test.head()


# ====================================================
# inference
# ====================================================
    model = EfficientNet_b0(6, pretrained=False)
    states = [load_state(config.MODEL_DIR+f'/{config.model_name}_fold{fold}.pth') for fold in config.trn_fold]

    test_dataset = TestDataset(test, config, transform=get_test_transforms(config))
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, 
                             num_workers=config.num_workers, pin_memory=True)

    predictions = inference(model, states, test_loader, device)
    print(predictions)
# submission
    test['label'] = predictions.argmax(1)
    test[['image_id', 'label']].to_csv(config.OUTPUT_DIR+'submission.csv', index=False)
    test.head()
#%%
if __name__ == '__main__':
    config = CFG
    main(config)
# %%
