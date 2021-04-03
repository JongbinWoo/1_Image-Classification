#%%
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.transforms import Normalize
from tqdm import tqdm
import pandas as pd
import os
from torch.utils.data.dataloader import DataLoader

from torch.utils.data.dataset import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

#config
from config import get_config

#model
from model.model import DenseNet, EfficientNet_b0


SEED = 42
torch.manual_seed(SEED)
#%%
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
# %%
def main(config):
    test_dir = '/opt/ml/input/data/eval'
    # meta 데이터와 이미지 경로를 불러옵니다.
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]

    transform = A.Compose([
        A.CenterCrop(p=1, height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    dataset = TestDataset(image_paths, transform)

    loader = DataLoader(
        dataset,
        shuffle=False
    )

    gender_age_model = DenseNet(6, config.MODEL.HIDDEN)
    mask_model = EfficientNet_b0(3, config.MODEL.HIDDEN)

    print('Load Test Model...\n')
    gender_age_checkpoint = torch.load(config.PATH.TEST_1)
    mask_checkpoint = torch.load(config.PATH.TEST_2)

    gender_age_model.load_state_dict(gender_age_checkpoint['model_state_dict'])
    mask_model.load_state_dict(mask_checkpoint['model_state_dict'])

    gender_age_model.to(config.SYSTEM.DEVICE)
    mask_model.to(config.SYSTEM.DEVICE)
    gender_age_model.eval()
    mask_model.eval()

    print('test inference start!')

    all_predictions = []
    for images in tqdm(loader):
        with torch.no_grad():
            images = images.to(config.SYSTEM.DEVICE)
            gender_age = gender_age_model(images).argmax(dim=-1)
            mask = mask_model(images).argmax(dim=-1)
            pred = mask * 6 + gender_age
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
    print('test inference is done!')



def test_visualization(model, test_loader, config):
    mnist_test = test_loader.dataset()

    n_samples = 64
    sample_indices = np.random.choice(len(mnist_test.targets), n_samples, replace=True)
    test_x = mnist_test.data[sample_indices]
    test_y = mnist_test.targets[sample_indices]

    with torch.no_grad():
        y_pred = model.forward(test_x.view(-1, 28*28).type(torch.float).to(config.DEVICE))
    
    y_pred = y_pred.argmax(axis=1)

    plt.figure(figsize=(20,20))
    
    for idx in range(n_samples):
        plt.subplot(8, 8, idx+1)
        plt.imshow(test_x[idx], cmap='gray')
        plt.axis('off')
        plt.title(f'Predict: {y_pred[idx]}, Label: {test_y[idx]}')
    plt.show()
    
# %%
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='MLP')
    # parser.add_argument('--r', default=None, type=str,
    #                     help='Path to checkpoint')
    # parser.add_argument('--batch_size', default=256, type=int)

    # args = parser.parse_args()

    config = get_config()

    main(config)