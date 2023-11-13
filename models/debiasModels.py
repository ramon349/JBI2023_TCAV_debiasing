import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from models.reverse_func import grad_reverse
from collections import OrderedDict
def remove_module(weights): 
    """" Removes the first 7 character of weight names. 
    Required since dataParallel  adds extra text due to messy saving. 
    TODO: Modify  the saving of model 

    """
    new_dict = OrderedDict()
    for e in weights.keys(): 
        new_keys = e[7:]
        new_dict[new_keys] = weights[e]
    return new_dict

class resnet18Bias(torch.nn.Module): 
    def __init__(self,num_task_classes,num_dem_classes): 
        ##TODO should the lambda be part of the model itself i don't think so 
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        in_features = self.model.fc.in_features
        self.fc= nn.Linear(in_features,num_task_classes)
        self.dem_fc = nn.Linear(in_features,num_dem_classes)
        self.debias = False 
    def forward(self,x):
        #Do the traditional forward of the resnet model 
        x = self.embed(x)
        #use the hidden representation and run it through both the branches 
        task_pred = self.fc(x) 
        if self.debias:
            dem_pred = self.dem_fc(grad_reverse(x))
        else: 
            dem_pred = self.dem_fc(x) 
        return task_pred,dem_pred
    def embed(self,x): 
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    def set_reversal(self,mode): 
        self.debias = mode 

class denseNet121Bias(torch.nn.Module): 
    def __init__(self, num_task_classes,num_dem_classes):
        super().__init__()
        self.model = models.densenet121(pretrained=True) 
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_task_classes)
        self.model.dem_classifier = nn.Linear(in_features,num_dem_classes) 
        self.debias=  False 
    def forward(self, x): 
        feats = self.model.features(x)
        out = F.relu(feats, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)  
        if self.debias: 
            task_classi = self.model.classifier(out) 
            dem_classi = self.model.dem_classifier(grad_reverse(out) )
        else: 
            task_classi = self.model.classifier(out) 
            dem_classi = self.model.dem_classifier(out) 
        return  task_classi,dem_classi
    def embed(self,x):
        feats = self.model.features(x)
        out = F.relu(feats, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1) 
        return out
    def set_reversal(self,mode): 
        self.debias = mode 