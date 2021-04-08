#%%
import os
from utils import seed_everything
from model.model import EfficientNet_b0,EfficientNet_b4, EfficientNet_mask
from dataloader.dataset import TestDataset
from dataloader.dataset import get_test_transforms
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import pickle
import warnings 
warnings.filterwarnings('ignore')

class CFG:
    debug=False
    num_workers=5
    model_name='F1_Fold' #0_ef0_ns.pth'
    size=512
    batch_size=128
    seed=2020
    target_size=5
    target_col='label'
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    ensemble=[2,10]
    inference=True
    OUTPUT_DIR = './'
    MODEL_DIR = '/opt/ml/1_Image-Classification/checkpoint'
    TEST_PATH = '/opt/ml/input/data/eval'
    MASK_PATH = '/opt/ml/code/checkpoint/EfficientNet_10'
# ====================================================
# Helper functions
# ====================================================
def load_state(model_path):
    # model = EfficientNet_b0(6, pretrained=False)
    try:  # single GPU model_file  
        state_dict = torch.load(model_path)
        # model.load_state_dict(state_dict, strict=True)
    except:  # multi GPU model_file
        state_dict = torch.load(model_path)
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}

    return state_dict

def load_state_(model_path):
    # model = EfficientNet_b0(6, pretrained=False)
    try:  # single GPU model_file  
        state_dict = torch.load(model_path)['model_state_dict']
        # model.load_state_dict(state_dict, strict=True)
    except:  # multi GPU model_file
        state_dict = torch.load(model_path)['model_state_dict']
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}

    return state_dict

def inference(model, states, test_loader, device):
    model.to(device)
    probs = []
    for images in tqdm(test_loader):
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
            avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
            # print(f'avg_preds: {avg_preds}')
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
        # print(f'probs: {probs}')
    probs = np.concatenate(probs)
    return probs


def main(config):
    seed_everything(seed=config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    test_dir = '/opt/ml/input/data/eval'
    image_dir = os.path.join(test_dir, 'images')
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]

    test_dataset = TestDataset(image_paths, transform=get_test_transforms(config))
    test_loader = DataLoader(test_dataset, 
                             batch_size=config.batch_size, 
                             shuffle=False, 
                             num_workers=config.num_workers, 
                             pin_memory=True)
    ##############    MASK     ##########################
    # model = EfficientNet_b0(3, pretrained=False)
    # states = [load_state(f'/opt/ml/1_Image-Classification/checkpoint/F1_Fold{fold}_mask.pth') 
    #                      for fold in range(4)]
                         
    # predictions = inference(model, states, test_loader, device)
    # submission['mask'] = predictions.argmax(1)
    # with open(os.path.join(config.OUTPUT_DIR, 'mask'), 'wb') as f:
    #     pickle.dump(predictions, f)
    ############## AGE / GENDER #########################
    model = EfficientNet_b0(6, pretrained=False)
    states = [load_state(f'/opt/ml/1_Image-Classification/checkpoint/F1_Fold{fold}_ef_final.pth') 
                         for fold in config.trn_fold]
    predictions_1 = inference(model, states, test_loader, device)
    with open(os.path.join(config.OUTPUT_DIR, 'final'), 'wb') as f:
        pickle.dump(predictions_1, f)
    ###### b0 ######
    # model = EfficientNet_b0(6, pretrained=False)
    # states = [load_state(f'/opt/ml/1_Image-Classification/checkpoint/F1_Fold{fold}_ef0_ns.pth') 
    #                      for fold in config.trn_fold]

    # predictions_0 = inference(model, states, test_loader, device)
    # with open(os.path.join(config.OUTPUT_DIR, '0'), 'wb') as f:
    #     pickle.dump(predictions_0, f)
    
    # ###### b4 ######
    # model = EfficientNet_b4(6, pretrained=False)
    # states = [load_state(f'/opt/ml/1_Image-Classification/checkpoint/F1_Fold{fold}_ef4_ns.pth') 
    #                      for fold in config.trn_fold]

    # predictions_4 = inference(model, states, test_loader, device)
    # with open(os.path.join(config.OUTPUT_DIR, '4'), 'wb') as f:
    #     pickle.dump(predictions_4, f)
    # submission['gender_age'] = (predictions_0*0.2+predictions_1*0.3+predictions_4*0.5).argmax(1)

    
    # def labeling(gender_age, mask):
    #     return mask * 6 + gender_age
    # submission['ans'] = submission.apply(lambda x: labeling(x['gender_age'], x['mask']), axis=1)

    # submission[['ImageID', 'ans']].to_csv(os.path.join(config.TEST_PATH, 'submission.csv'), index=False)
    # submission.head()

#%%
if __name__ == '__main__':
    config = CFG
    main(config)
# %%
# pred = mask * 6 + gender_age
# def concat_gender_age(gender, age):
#     return gender * 3 + age

# train_df['gender_age_class'] = train_df.apply(lambda x: concat_gender_age(x['gender'], x['age_class']), axis=1)


# train_df.to_csv('/opt/ml/input/data/train/train.csv')
