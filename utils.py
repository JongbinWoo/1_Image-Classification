from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import random 
import os

import torchvision 

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg.astype(np.uint8), (1,2,0)))

def images_to_probs(net, images):
    output = net(images)
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(net, images, labels):
    preds, probs = images_to_probs(net, images)
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(len(labels)):
        ax = fig.add_subplot(8, 8, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=False)
        ax.set_title(f'{preds[idx]}, {probs[idx]:.1f}%\n(label: {labels[idx]})', 
                     color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

#%%
def find_nonzero_index(inputs):
    """
    inputs: (N) Tensor
    outputs: (n) list
    """
    return torch.nonzero(inputs, as_tuple=True)[0].tolist()

def convert_target(num_class, targets, ages):
    targets = F.one_hot(targets, num_classes=num_class).float()
    ages_index_1 = find_nonzero_index(torch.logical_and((50 < ages), (ages < 60)))
    for idx in ages_index_1:
        cls_idx = find_nonzero_index(targets[idx])[0]

        targets[idx][cls_idx] -= (ages[idx]-50)*0.05
        targets[idx][cls_idx+1] += (ages[idx]-50)*0.05
    # print(targets)
    ages_index_2 = find_nonzero_index((ages == 60))
    for idx in ages_index_2:
        cls_idx = find_nonzero_index(targets[idx])[0]

        targets[idx][cls_idx] -= 0.276
        targets[idx][cls_idx-1] += 0.276

    ages_index_3 = find_nonzero_index(torch.logical_and((24 < ages), (ages < 30)))
    for idx in ages_index_3:
        cls_idx = find_nonzero_index(targets[idx])[0]

        targets[idx][cls_idx] -= (ages[idx]-20)*0.05
        targets[idx][cls_idx+1] += (ages[idx]-20)*0.05
    
    ages_index_4 = find_nonzero_index(torch.logical_and((29 < ages), (ages < 35)))
    for idx in ages_index_4:
        cls_idx = find_nonzero_index(targets[idx])[0]

        targets[idx][cls_idx] -= (39-ages[idx])*0.05
        targets[idx][cls_idx-1] += (39-ages[idx])*0.05
    
    return targets

#%%
def show_images(inputs):
    inputs -= inputs.min()
    inputs /= inputs.max()
    plt.imshow(torchvision.utils.make_grid(inputs.cpu(), nrow=5).permute(1, 2, 0))
    plt.savefig('./foo.png')
    plt.show()

#%%
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


