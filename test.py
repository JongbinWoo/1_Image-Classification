#%%
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.transforms import Normalize
from tqdm import tqdm
import pandas as pd
import os
from torch.utils.data.dataloader import DataLoader

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, CenterCrop


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
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
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

    transform = transforms.Compose([
        CenterCrop(224),
        ToTensor(),
        Normalize((0.560, 0.524, 0.501), (0.233, 0.243, 0.245))
    ])
    dataset = TestDataset(image_paths, transform)

    loader = DataLoader(
        dataset,
        shuffle=False
    )

    model = EfficientNet_b0(config.DATASET.NUM_CLASSES, 512)
    checkpoint = torch.load(config.PATH.RESUME)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(config.SYSTEM.DEVICE)
    model.eval()

    print('test inference start!')

    all_predictions = []
    for images in tqdm(loader):
        with torch.no_grad():
            images = images.to(config.SYSTEM.DEVICE)
            pred = model(images)
            # pred = pred.argmax(dim=-1)
            pred = [p.argmax(dim=-1).cpu().numpy() for p in pred]
            all_predictions.extend(pred[0]*3 + pred[1] + pred[2] *6)
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