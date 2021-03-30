import torch.optim as optim

def get_optimizer(optimizer_name='Adam', lr=0.001, model=None):
    optimizer =  optim.__dict__[optimizer_name]
    params = [p for p in model.parameters() if p.requires_grad]
    return optimizer(params, lr=lr)

