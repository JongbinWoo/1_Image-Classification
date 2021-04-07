#%%
# from dataloader.dataset import get_all_split_loaders
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys 
from glob import glob
import cv2
import albumentations as A
# #%%
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
#%%
train_df.age.value_counts().sort_index().T

#%%
transform = A.Compose([
    A.CenterCrop(p=1, height=384, width=256)
])
#%%
image_batch = []
image_batch_labels = []
data_root = '/opt/ml/input/data/train/images'
def visualize_all_images():
    """
    subplot으로 했을떄 에러가 나서 수정중....
    """
    
    folder_list = glob(os.path.join(data_root, '*_50'))
    for j in range(1):
        fig, ax = plt.subplots(5,5, figsize=(15, 15))
        for i in range(25):
            
            folder_path = folder_list[j*25 + i]
            img_path = glob(os.path.join(folder_path, 'normal.jpg'))[0]
            
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed = transform(image=image)
            transformed_image = transformed['image']
            image_batch.append(transformed_image)
            image_batch_labels.append([0.5, 0.5])
            ax[i//6][i%5].imshow(transformed_image[:, :128])
        plt.show()


#%%
visualize_all_images()
# train_dataset = MaskDataset(data_root)   
#%%
image_batch = np.array(image_batch)
image_batch_labels = np.array(image_batch_labels)
# %%
# fig, ax = plt.subplots(2,5, figsize=(20, 10))
# images = train_batch[0]
# for i,image in enumerate(images):
#     # image = image.transpose(0,2)
#     if i < 5:
#         ax[0, i].imshow(image)
#     else:
#         ax[1, i%5].imshow(image)
# plt.show()

#%%
def generate_cutmix_image(image_batch, image_batch_labels, beta):
    """ Generate a CutMix augmented image from a batch 
    Args:
        - image_batch: a batch of input images
        - image_batch_labels: labels corresponding to the image batch
        - beta: a parameter of Beta distribution.
    Returns:
        - CutMix image batch, updated labels
    """
    # generate mixed sample
    lam = 0.5 #np.random.beta(beta, beta)

    rand_index = np.random.permutation(len(image_batch))
    target_a = image_batch_labels
    target_b = image_batch_labels[rand_index]
    # bbx1, bby1, bbx2, bby2 = rand_bbox(image_batch[0].shape, lam)
    image_batch_updated = image_batch.copy()
    image_batch_updated[:, :, :128, :] = image_batch[rand_index, :, :128, :]
    
    # adjust lambda to exactly match pixel ratio
    # lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image_batch.shape[1] * image_batch.shape[2]))
    label = target_a * lam + target_b * (1. - lam)
    
    return image_batch_updated, label
# %%
image_batch_updated, image_labels_updated = generate_cutmix_image(image_batch, image_batch_labels, 1.0)
# %%
# Show CutMix images
print("CutMix Images")
for i in range(2):
    for j in range(2):
        plt.subplot(2,2,2*i+j+1)
        plt.imshow(image_batch_updated[2*i+j])
plt.show()
# %%
