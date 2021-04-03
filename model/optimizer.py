import torch.optim as optim
from adamp import AdamP

def get_optimizer(optimizer_name='Adam', lr=0.001, model=None):
    optimizer =  optim.__dict__[optimizer_name]
    params = [p for p in model.parameters() if p.requires_grad]
    return optimizer(params, lr=lr)

def get_adamp(lr=0.001, model=None, weight_decay=1e-6):
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamP(params, lr=lr)