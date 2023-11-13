import pdb 
import torch
from torchvision import models as torch_models 
from models.debiasModels import resnet18Bias,denseNet121Bias
from torch import nn 

def remove_module(weights,pattern): 
    from collections import OrderedDict
    new_dict = OrderedDict()
    for e in weights.keys(): 
        new_keys = e.replace(pattern,"")
        new_dict[new_keys] = weights[e]
    return new_dict
        
def model_builder(
    model_name="densenet",
    model_weight="",
    config=None 
):
    model = None 
    num_task = config['num_task_classes'] 
    ablate_layer = config['freeze_layer'] # If not being used should be the empty string
    if model_name == "densenet":
        model = torch_models.densenet121("DEFAULT")
        in_feats = model.classifier.in_features
        model.classifier = nn.Linear(in_features=in_feats,out_features=num_task)
    if model_name =='resnet': 
        model =torch_models.resnet18("DEFAULT") 
        in_feats = model.fc.in_features 
        model.fc = nn.Linear(in_features=in_feats,out_features=num_task)
    if model_name == 'debiasDensenet': 
        model =  denseNet121Bias(config['num_task_classes'],config['dem_num_classes'])
        freeze_upto(model,ablation_layer=ablate_layer)
    if model_name == 'debiasResnet': 
        model =  resnet18Bias(config['num_task_classes'],config['dem_num_classes'])
        freeze_upto(model,ablation_layer=ablate_layer)
    if len(model_weight) >0:
        print("loading checkpoint")
        ck = torch.load(model_weight, map_location="cpu")
        state_dict = ck['model_state_dict']
        state_dict =  remove_module(state_dict,"module.") 
        state_dict =  remove_module(state_dict,"_orig_mod.") 
        model.load_state_dict(state_dict)
    else: 
        print("din't load any weights")
    return model

def find_layer(layer_name,all_layer_names): 
    for i,e in enumerate(all_layer_names): 
        if e.startswith(layer_name): 
            return i 
    raise ValueError(f"{layer_name} was not found")
def freeze_upto(model,ablation_layer): 
    """  Given the name of a weight parameter we freeze all layers until we find it
    """
    #get a list of all the parameters 
    if ablation_layer: 
        names = [n for n,e in model.named_parameters()]  
        idx = find_layer(ablation_layer,names)
        for i,e in enumerate(model.parameters()):
            if i <  idx: 
                e.requires_grad=False 
if __name__ == "__main__":
    model = model_builder(model_name='densenet')