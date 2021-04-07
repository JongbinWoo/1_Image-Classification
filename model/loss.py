#%%
from re import M
import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
class CustomLoss(nn.Module):
    def __init__(self, T=10):
        super(CustomLoss, self).__init__()
        # self.weight = weight.unsqueeze(0)
        self.T = T
    
    def forward(self, outputs, targets):
        """
        outputs: (B, C)
        targets: (B)
        ages: (B)
        """
        # 
        # soft_targets = convert_target(outputs, targets, ages)
        logsoftmax = nn.LogSoftmax(dim=-1)

        # loss = torch.mean(torch.sum(-soft_targets * logsoftmax(outputs / self.T) * self.weight, 1))
        loss2 = torch.mean(torch.sum(-targets * logsoftmax(outputs / self.T), 1))
        return loss2
#%%
# implementations reference - https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/taylor_softmax.py
# paper - https://www.ijcai.org/Proceedings/2020/0305.pdf

class TaylorSoftmax(nn.Module):

    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n+1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out

class LabelSmoothing(nn.Module):

    def __init__(self, classes, smoothing=0.0, dim=-1): 
        super(LabelSmoothing, self).__init__() 
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing 
        self.cls = classes 
        self.dim = dim 
    def forward(self, batch_size, target): 
        """Taylor Softmax and log are already applied on the logits"""
        #batch_size = batch_size.log_softmax(dim=self.dim) 
        with torch.no_grad(): 
            true_dist = torch.zeros(batch_size, self.cls).cuda()
            true_dist.fill_(self.smoothing / (self.cls - 1)) 
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        return true_dist


class TaylorCrossEntropyLoss(nn.Module):

    def __init__(self,n=2, ignore_index=-1, reduction='mean', smoothing=0.2):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        # self.reduction = reduction
        # self.ignore_index = ignore_index
        # self.lab_smooth = LabelSmoothingLoss(6, smoothing=smoothing)
        # self.loss = CustomLoss()
        
    def forward(self, logits, soft_labels):

        log_probs = self.taylor_softmax(logits).log()
        #loss = F.nll_loss(log_probs, labels, reduction=self.reduction,
        #        ignore_index=self.ignore_index)
        # loss = self.lab_smooth(log_probs, labels)
        # loss = self.loss(log_probs, labels)
        return torch.mean(torch.sum(-soft_labels * log_probs, dim=-1))
        
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )




# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    def __init__(self, classes=3, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        # assert y_true.ndim == 1
        # y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()
# # %%
# inputs = torch.randn(6, 6)
# labels = torch.tensor([0, 1, 2, 3, 4, 5])
# tce = TaylorCrossEntropyLoss()
# tce(inputs, labels)

#https://github.com/yuanli2333/Teacher-free-Knowledge-Distillation/blob/master/my_loss_function.py
class loss_kd_regularization(nn.Module):
    
    def __init__(self, config):
        super(loss_kd_regularization, self).__init__()
        self.alpha = config.TRAIN.ALPHA
        self.T = config.TRAIN.T
        self.M = config.TRAIN.M
        self.taylor_CE = TaylorCrossEntropyLoss()
    
    def forward(self, logits, soft_targets):
        loss_TCE = self.taylor_CE(logits, soft_targets)
        loss_soft_reg = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits/self.T, dim=1), F.softmax(soft_targets, dim=1))

        KD_loss = (1. - self.alpha)*loss_TCE + self.alpha*loss_soft_reg
        return KD_loss


class Cumbo_Loss(nn.Module):
    def __init__(self, config):
        super(Cumbo_Loss, self).__init__()
        self.kd_reg = loss_kd_regularization(config)
        self.f1_loss = F1Loss(6)
        self.gamma = config.TRAIN.GAMMA

    def forward(self, logits, soft_labels):
        kd_reg = self.kd_reg(logits, soft_labels)
        f1_loss = self.f1_loss(logits, soft_labels)
        return (1. - self.gamma) * kd_reg + self.gamma * f1_loss 