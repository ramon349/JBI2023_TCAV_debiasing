import pandas as pd 
import sys 
sys.path.append('..')
import pandas as pd
import numpy as np
from collections import OrderedDict
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from torch.utils.data.dataloader import DataLoader
from plain_model_dist import model_builder,remove_module
from captum.attr import  LayerIntegratedGradients 
from captum.concept import TCAV
from captum.concept import Concept
import torch
torch.multiprocessing.set_sharing_strategy('file_system') 

from types import MethodType
import torch.nn.functional as F 
import matplotlib.pyplot as plt
from loader_factory import get_skin_loaders
from helper_funcs import build_transform
import pdb 
import sys
import json 
def get_truths(input_path,samples_per_concept=50):
    val_df =pd.read_csv(input_path)
    val_df = val_df[val_df['split']=='train'] 
    val_df = val_df[val_df['fitzpatrick'].isin([1,6])] 
    concept_test = val_df.sample(300,random_state=1996)
    rem_samples = val_df[~val_df['file'].isin(concept_test['file'])]
    black_df = rem_samples[rem_samples['fitzpatrick'].isin([6])].copy().sample(samples_per_concept)
    white_df = rem_samples[rem_samples['fitzpatrick'].isin([1]) ].copy().sample(samples_per_concept)
    #make sure samples are removed from the other dataset 
    rem_samples = rem_samples[~rem_samples['file'].isin(black_df['file'].unique()) ]
    rem_samples = rem_samples[~rem_samples['file'].isin(white_df['file'].unique()) ]
    rem_samples = rem_samples[rem_samples['fitzpatrick'].isin([1])]
    return black_df,white_df, rem_samples

#layer code 
def find_conv_layers(model):
    conv_layers= OrderedDict()
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            conv_layers[name] = layer 
    return conv_layers

def find_relu_layers(model):
    conv_layers= OrderedDict()
    for name, layer in model.named_modules():
        if isinstance(layer,  torch.nn.ReLU): 
            conv_layers[name] = layer 
    return conv_layers


def make_concept(df,id,concept_name,loader,transforms): 
    """ Instantiate a concept object 
    - Dataframe should contain images related to one concept  
    """
    ds =  loader(df,transform=transforms)
    data_loader = DataLoader(ds,batch_size=1,num_workers=16) 
    concept= Concept(id=id,name=concept_name,data_iter=data_loader)
    return concept

def make_concepts(black_df,white_df,my_loader,transforms):
    black_concept = make_concept(black_df,0,'Black',my_loader,transforms)
    white_concept = make_concept(white_df,1,'Others',my_loader,transforms)
    return black_concept,white_concept

def main(): 
    with open(sys.argv[1],'r') as f:  
        input_config = json.load(f)
    data_path = input_config['data_path']
    weight_path = input_config['weight_path']
    samples_per_concept = input_config['samples_per_concept']
    black_df,white_df,rem_df = get_truths(data_path,samples_per_concept=samples_per_concept) 
    #i use a custom dataloader that only returns the image 
    skinLoader = get_skin_loaders('single')
    DEVICE = 'cpu'
    _,val_transform = build_transform(input_config)
    test_ds = skinLoader(rem_df,transform=val_transform)
    (black_c,white_c) = make_concepts(black_df,white_df,skinLoader,val_transform)
    model =  model_loading('densenet',w_path=weight_path)
    cat = 'conv'
    if cat =='relu': 
        relu_layers = find_relu_layers(model)
        layers_interest_names = list(relu_layers.keys())
    if cat =='conv':
        layers_interest = find_conv_layers(model)
        layers_interest_names = list(layers_interest.keys())
    names = layers_interest_names[::10] #TODO: for viz purpose i cut down on the number of  layers 
    zebra_tensors = torch.stack([test_ds.__getitem__(idx).to(DEVICE) for idx  in range(25)])
    model.eval()
    model.forward  = MethodType(forward, model)
    mytcav = TCAV(model=model,
                layers=names,
                save_path=f'./cav_eval_test',
                layer_attr_method=LayerIntegratedGradients(
                    model, None)
    )
    tcav_scores_w_random = mytcav.interpret(inputs=zebra_tensors,
                                            experimental_sets=[[black_c,white_c]],
                                            processes=6,
                                            target=(1,),
                                            internal_batch_size=8,
                                            n_steps=20,
                                        )
    acc_scores,layer_names =  get_accuracy_scores(mytcav) 
    layer_names = [".".join(e.split('.')[-2: ]) for e in layer_names]
    fig = plt.figure(dpi=300)
    plt.plot(np.hstack(acc_scores)) 
    plt.xticks(range(0,len(acc_scores)),layer_names,rotation=90)
    plt.ylabel('TCAV Accuracy')
    plt.title('TCAV plot Densenet fitzpatrick classification Accuracy')
    plt.tight_layout()
    plt.savefig('./results/figures/tcav_accuracy.png')
    tcav_scores,layer_names = get_tcav_scores(tcav_scores_w_random)
    fig = plt.figure(dpi=300)
    plt.plot(np.hstack(tcav_scores),range(0,len(tcav_scores))) 
    plt.barh(range(0,len(tcav_scores)),np.hstack(tcav_scores),align='center')
    plt.yticks(range(0,len(tcav_scores)), labels=layer_names)
    plt.ylabel('Layer Name')
    plt.xlabel('Layer TCAV Score')
    plt.title('TCAV plot Densenet fitzpatrick')
    plt.tight_layout()
    plt.savefig('./results/figures/tcav_score.png')


def get_accuracy_scores(mytcav): 
    layer_names = list(mytcav.cavs['0-1'].keys())
    my_actual_scores = [ mytcav.cavs['0-1'][e].stats['accs'] for e in layer_names]
    layer_names_s = [e.split('features')[1] for e in layer_names]
    return (my_actual_scores,layer_names_s)
def get_tcav_scores(cav_interp): 
    scores = list() 
    names =list()
    for e in cav_interp['0-1'].keys():
        names.append(e)
        scores.append(cav_interp['0-1'][e]['sign_count'][1].cpu().numpy())
    return scores,names

def forward(self,x): 
        #feats = self.model.features(x.to('cuda:1'))
        feats = self.features(x )
        out = F.relu(feats, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        task_pred = self.classifier(out)
        return task_pred.cpu()

def model_loading(model_name,w_path=""):
    model = model_builder(model_name,model_weight=w_path,config={"num_task_classes":2,'freeze_layer':""})
    weights = torch.load(w_path,map_location='cpu')
    weights['model_state_dict']  = remove_module(weights['model_state_dict'],"model.") 
    weights['model_state_dict']  = remove_module(weights['model_state_dict'],"module.") 
    model.load_state_dict(weights['model_state_dict'])
    for e in model.parameters():
        e.requires_grad = True
    return model 

if __name__=='__main__':
    main()
