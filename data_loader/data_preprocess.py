#%%
import pandas as pd

import os
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

folder_names = train_df.path.values
data_root = '/opt/ml/input/data/train/images'
# %%
# Data Preprocessing

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
