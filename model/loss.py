#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
def find_nonzero_index(inputs):
    """
    inputs: (N) Tensor
    outputs: (n) list
    """
    return torch.nonzero(inputs, as_tuple=True)[0].tolist()
# %%
def convert_target(inputs, targets, ages):
    targets = F.one_hot(targets, num_classes=inputs.size()[-1]).float()
    ages_index = find_nonzero_index(torch.logical_and((50 < ages), (ages < 60)))
    for idx in ages_index:
        cls_idx = find_nonzero_index(targets[idx])[0]

        targets[idx][cls_idx] -= (ages[idx]-50)*0.1
        targets[idx][cls_idx+1] += (ages[idx]-50)*0.1
    return targets
#%%
class CustomLoss(nn.Module):
    def __init__(self, weight, T):
        super(CustomLoss, self).__init__()
        self.weight = weight.unsqueeze(0)
        self.T = T
    
    def forward(self, outputs, targets, ages):
        """
        outputs: (B, C)
        targets: (B)
        ages: (B)
        """
        # 
        soft_targets = convert_target(outputs, targets, ages)
        logsoftmax = nn.LogSoftmax()

        loss = torch.mean(torch.sum(-soft_targets * logsoftmax(outputs / self.T) * self.weight, 1))
        return loss
#%%
# inputs = torch.randn(1, 6)
# ages = torch.tensor([40, 50, 53, 55, 59, 60])
# targets = torch.tensor([1,1,1,2,2,3])
# soft_targets = convert_target(inputs, targets, ages)
# # %%
# loss = CustomLoss(torch.tensor([1, 1, 1, 1, 1, 1]))
# # %%
# loss(inputs, targets, ages)


# %%
