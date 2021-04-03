#%%
from torchvision import models
import torch.nn as nn 
import torch.nn.functional as F
import torch

#%%
# Pretrained_model


model_name = 'densenet'
num_classes = [2, 3, 3]
feature_extract = True 

#%%
def set_parameter_requires_grad(model, freeze):
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

#%%
class DenseNet(nn.Module):
    def __init__(self, num_classes, hidden_dim, freeze=True):
        super(DenseNet, self).__init__()
        pretrained_model = models.densenet121(pretrained=True, progress=False)
        print('Loaded pretrained weightes for DenseNet121\n')
        # pretrained_model = models.vgg16_bn(pretrained=True, progress=False)
        self.feature_extractor = nn.Sequential(*(list(pretrained_model.children())))[:-1]
        set_parameter_requires_grad(self.feature_extractor, freeze)

        # Three Classifiers
        self.num_features = pretrained_model.classifier.in_features 
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.fc = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_dim),
            self.relu,
            self.dropout,
            nn.Linear(self.hidden_dim, num_classes))
        
            
            
        
        # initialize all fc layers to xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain = 1)
                
    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def __str__(self):
        num_parameters = 0
        for params in self.parameters():
            if params.requires_grad:
                num_parameters += len(params.reshape(-1))
        
        return super().__str__() + f'\n\nNumber of Trainable Parameters: {num_parameters:,d}\n'

# %%
# input_sample = torch.rand(1, 3, 224, 224)

# from torchsummary import summary
# model = models.densenet121(pretrained=True, progress=False)
# summary(model.cuda(), (3, 224, 224))
# %%
from efficientnet_pytorch import EfficientNet
class EfficientNet_b0(nn.Module):
    def __init__(self, num_classes, hidden_dim, freeze=True):
        super(EfficientNet_b0, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        set_parameter_requires_grad(self.model, True)

        # Three Classifiers
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.fc = nn.Sequential(
            nn.Linear(1280, self.hidden_dim),
            self.relu,
            self.dropout,
            nn.Linear(self.hidden_dim, num_classes))
        
        
            
        
        # initialize all fc layers to xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain = 1)
                
    def forward(self, x):
        x = self.model.extract_features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def __str__(self):
        num_parameters = 0
        for params in self.parameters():
            if params.requires_grad:
                num_parameters += len(params.reshape(-1))
        
        return super().__str__() + f'\n\nNumber of Trainable Parameters: {num_parameters:,d}\n'




# %%
# model = EfficientNet.from_pretrained('efficientnet-b0')
# %%
