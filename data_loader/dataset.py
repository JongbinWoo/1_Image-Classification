#%%
from torch.utils import data
from torch.utils.data import Dataset 
from torchvision import transforms
import pandas as pd
import cv2
import os 
from glob import glob 



#%%
class MaskDataset(Dataset):
    def __init__(self, data_root, transform=None):
        super(MaskDataset, self).__init__()
        self.info_df = pd.read_csv('/opt/ml/input/data/train/train.csv')
        self.folder_list = self._get_folder_list()
        self.len = len(self.folder_list) * 7
        self.data_root = data_root

        self.transform = transform
               

    def __getitem__(self, index):
        folder_idx, img_idx = divmod(index, 7)
        folder_path = self.folder_list[folder_idx]
        folder_path = os.path.join(data_root, folder_path)
        
        img_list = glob(os.path.join(folder_path, '*.jpg'))
        img_list = list(map(os.path.basename, img_list))
        img_name = img_list[img_idx]
        
        if 'normal' in img_name: 
            mask = 2
        elif 'incorrect_mask' in img_name: 
            mask = 1
        else:
            mask = 0
        
        gender, age = self._get_class(folder_idx)
        
        img_path = os.path.join(folder_path, img_name)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255.

        if self.transform:
            img = self.transform(img)
        
        
        return img, [gender, age, mask]

    def __len__(self):
        return self.len
    
    def _get_folder_list(self):
        return self.info_df.path.values

    def _get_class(self, idx):
        idx_data =  self.info_df.iloc[idx]
        return idx_data.gender, idx_data.age_class
     
# %%
data_root = '/opt/ml/input/data/train/images'

train_dataset = MaskDataset(data_root, transform=transforms.ToTensor())

# %%
len(train_dataset)

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(len(train_dataset), 7)
for j, folder_name in enumerate((train_dataset.folder_list)):
    folder_path = os.path.join(data_root, folder_name)
    
    img_path_list = glob(os.path.join(folder_path, '*.jpg'))
    img_list = list(map(os.path.basename, img_path_list))

    for i in range(len(img_path_list)):
        
        img = train_dataset[j*7+i][0].detach().cpu().numpy()
        # print(len(img))
        img = img.transpose(1, 2, 0)
        axes[j, i].imshow(img)
        axes[j, i].axis('off')
        axes[j, i].title(img_list[i])
    
# %%
