#%%
from torchvision import models
import torch.nn as nn 
import torch

#%%
# Pretrained_model


model_name = 'densenet'
num_classes = [2, 3, 3]
feature_extract = True 

#%%
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

#%%
class VGG(nn.Module):
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        # pretrained_model = models.densenet121(pretrained=True, progress=False)

        pretrained_model = models.vgg16_bn(pretrained=True, progress=False)
        self.feature_extractor = nn.Sequential(*(list(pretrained_model.children())))[:-2]
        set_parameter_requires_grad(self.feature_extractor, True)

        # Three Classifiers
        self.num_features = pretrained_model.classifier[0].in_features
        
        self.gender_classifier = nn.Linear(self.num_features, num_classes[0])
        self.age_classifier = nn.Linear(self.num_features, num_classes[1])
        self.mask_classifier = nn.Linear(self.num_features, num_classes[2])

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        gender_class = self.gender_classifier(x)
        age_class = self.age_classifier(x)
        mask_class = self.mask_classifier(x)
        return gender_class, age_class, mask_class

# %%
vgg = VGG(num_classes)
# %%
input_sample = torch.randn(1, 3, 224, 224)
vgg(input_sample)
# %%
