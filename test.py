#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse

#data
from data_loader.data_loader import get_loader
from data_loader.dataset import get_mnist, get_augmentation

#config
from config import get_config

#model
from model.model import MLP
from model import loss
from model.optimizer import get_optimizer 

SEED = 42
torch.manual_seed(SEED)

# %%
def main(config):
    transform = get_augmentation(**config.TRAIN.AUGMENTATION)
    _, test_dataset = get_mnist(config.PATH.ROOT, transform=transform)


    test_loader = get_loader(test_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True)
    
    mlp = MLP(input_features=784, hidden_size=256, output_features=config.DATASET.NUM_CLASSES)
    checkpoint = torch.load(config.PATH.CHECKPOINT)
    state_dict = checkpoint['state_dict']
    mlp.load_state_dict(state_dict)
    mlp.to(config.DEVICE)
    mlp.eval()

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
    parser = argparse.ArgumentParser(description='MLP')
    parser.add_argument('--r', default=None, type=str,
                        help='Path to checkpoint')
    parser.add_argument('--batch_size', default=256, type=int)

    args = parser.parse_args()

    config = get_config(args)

    main(config)