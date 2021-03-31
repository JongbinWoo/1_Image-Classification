#%%
import matplotlib.pyplot as plt 
# import seaborn as sns
import pandas as pd

import os
import sys 
from glob import glob

#%%
# Load data
train_df = pd.read_csv('/opt/ml/input/data/train/train.csv')
train_df = train_df[['id', 'gender', 'age', 'path']]
# %%
# Set classes of each image
train_df.gender.replace(
    {'male': 0, 'female':1}, inplace=True
)

def _get_age_class(age):
    if age >= 60:
        return 2
    elif 30 <= age < 60:
        return 1
    else:
        return 0

train_df['age_class'] = train_df.apply(lambda x: _get_age_class(x['age']), axis=1)

#%%
def concat_gender_age(gender, age):
    return gender * 3 + age

train_df['gender_age_class'] = train_df.apply(lambda x: concat_gender_age(x['gender'], x['age_class']), axis=1)


train_df.to_csv('/opt/ml/input/data/train/train.csv')
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

# 나이를 어떻게하면 잘 구분할까. 이런 문제를 찾아보자.







# %%
# Data Preprocessing

folder_names = train_df.path.values
data_root = '/opt/ml/input/data/train/images'

# %%
def check_num_files(root, folder_names):
    for folder_name in folder_names:
        path = os.path.join(root, folder_name)
        img_list = glob(os.path.join(path, '*.jpg'))
        img_list = list(map(os.path.basename, img_list))

        if len(img_list) != 7:
            print(img_list)
# %%
check_num_files(data_root, folder_names)
#%%
from PIL import Image

def png2jpg(root, folder_names):
    """
    png -> jpg, jpeg -> jpg
    """
    for folder_name in folder_names:
        path = os.path.join(root, folder_name)
        img_list = glob(os.path.join(path, '*.png')) # .jpeg
        # img_list = list(map(os.path.basename, img_list))

        for img_path in img_list:
            # print(img_path)
            img = Image.open(img_path)
            new_path = img_path[:-4]+'.jpg' # 5
            img.save(new_path)

png2jpg(data_root, folder_names)

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
