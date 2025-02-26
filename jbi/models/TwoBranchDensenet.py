
from torchvision.models import densenet121 ,DenseNet121_Weights
from torchvision.models.densenet import DenseNet ,_load_state_dict
from torch import nn 
from torch.nn import functional as F 
import torch 
from .model_factory import ModelRegister
from torch.autograd import Function

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        out = grad_output.neg()
        return out 


def grad_reverse(x):
    return GradReverse.apply(x)

@ModelRegister.register("DensenetTwoTask")
class Densenet121TwoBranch(DenseNet): 

    def _call_init(self): 
        "This makes the densenet121 init params" 
        growth_rate= 32 
        block_config = (6, 12, 24, 16)
        num_init_features= 64 
        weights = DenseNet121_Weights .IMAGENET1K_V1
        super().__init__(growth_rate=growth_rate,block_config=block_config,num_init_features=num_init_features)
        _load_state_dict(model=self,weights=weights,progress=True)
    def __init__(self, config): 
        self._call_init()  
        self.add_demo_classifier(config['num_demo'])
        self.mod_classifier(config['num_task'])

    def add_demo_classifier(self,num_demo): 
        in_features = self.classifier.in_features
        self.demo_classifier = nn.Linear(in_features,num_demo)
    def mod_classifier(self,num_task):
        in_features = self.classifier.in_features
        self.classifier = nn.Linear(in_features,num_task)

    def forward(self,x,task='all'):
        features = self.features(x)
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        if task =='all': 
            task_out = self.classifier(features)
            dem_out = self.demo_classifier(features)
            return task_out,dem_out
        if task =='demo': 
            dem_out = self.demo_classifier(features)
            return dem_out
        if task =='task': 
            task_out = self.classifier(features)
            return task_out
        
@ModelRegister.register("DensenetTwoTaskAdv")
class Densenet121Adv(Densenet121TwoBranch): 
    def __init__(self, config):
        super().__init__(config) 
    def forward(self, x, task='phase1'):
        features = self.features(x)
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        if task =='phase1':  #do this so the demographic prediciton only updates the  discriminator
            task_out = self.classifier(features)
            dem_out = self.demo_classifier(features.detach())
            return task_out,dem_out
        if task =='phase2': 
            task_out = self.classifier(features)
            dem_out = self.demo_classifier(self._reversal_layer(features))
            return task_out,dem_out
    def _reversal_layer(self,x):
        #doing this so we can also have the confusion lsos version be very similar
        return grad_reverse(x)

@ModelRegister.register("DensenetTwoTaskConf")
class Densenet121Conf(Densenet121Adv): 
    def __init__(self, config):
        super().__init__(config)
    def _reversal_layer(self, x):
        return x


if __name__ =="__main__":
    dummy_conf={"num_task":2,"num_demo":3}

    mdoel = Densenet121TwoBranch(config=dummy_conf)