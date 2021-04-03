#%%
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd

import os
import sys 
from glob import glob

#%%
# Load data
train_df = pd.read_csv('/opt/ml/input/data/train/train.csv')

# %%
# Plot
sns.countplot(x='gender', data=train_df)
#%%
sns.countplot(x='age_class', hue='gender', data=train_df)
#%%
# 나이의 경계에 있는 사람들...
sns.histplot(train_df['age'], bins=list(range(17,63)))
plt.axvline(30, color='red')
plt.axvline(60, color='red')

# #%%
# from data_loader.dataset import MaskDataset
# import matplotlib.pyplot as plt

# def visualize_all_images(dataset):
#     """
#     subplot으로 했을떄 에러가 나서 수정중....
#     """
#     fig, axes = plt.subplots(len(dataset), 7)
#     for j, folder_name in enumerate((dataset.folder_list)):
#         folder_path = os.path.join(data_root, folder_name)
        
#         img_path_list = glob(os.path.join(folder_path, '*.jpg'))
#         img_list = list(map(os.path.basename, img_path_list))

#         for i in range(len(img_path_list)):
            
#             img = dataset[j*7+i][0].detach().cpu().numpy()
#             # print(len(img))
#             img = img.transpose(1, 2, 0)
#             axes[j, i].imshow(img)
#             axes[j, i].axis('off')
#             axes[j, i].title(img_list[i])

# train_dataset = MaskDataset(data_root)   

# %%
