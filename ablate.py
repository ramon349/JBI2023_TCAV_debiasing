import pickle as pkl 
import torch 
import pandas as pd 
from sklearn.metrics import accuracy_score 
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score ,silhouette_score,accuracy_score,roc_auc_score 
from collections import OrderedDict
import numpy as np 
from plain_model_dist import model_builder
from data_iterators.batch_iterators import test_iterator
from testCode import test_bias_iterator 
from tqdm import tqdm 
import pdb 
def get_output(self,input,output): 
    #this is uses as a  layer hook to figure out what the output dimensionality of a layer is. 
    #There might be a better approach to figure out dimensions but i'm not sure 
    self.out_vec =  output
#similarity code 
def flatten_weights(ws):
    #helper method to  flatten weight's into vector for similairty comparison . 
    w_mat = list()
    for e in range(ws.shape[0]):
        vec = ws[e].reshape(1,-1).cpu()
        norm = torch.norm(vec)
        w_mat.append(vec/norm)
    return torch.cat(w_mat)

#layer code 
def find_conv_layers(model):
    conv_layers= OrderedDict()
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            conv_layers[name] = layer 
    return conv_layers

def get_layer_mask(model,layer,sample_input,perc):
    #figure out masking policy for a certain layer 
    # start by identifying similar weight groups 
    weights = layer.weight
    num_filts = weights.shape[0]
    flat =  flatten_weights(layer.weight)
    group_l = find_best_k(flat,mode='perc',perc=perc)
    return group_l

#change clustering to be the one from datamining 

def find_best_k(w_mat,mode='perc',perc=None):
    # we need to identify a consistent group of layer features 
    #this searches for clusters 2 through 10 
    #return the group labels for the layers and the number of groups 
    if mode=='perc':
        group_l = get_weight_2_drop(w_mat,perc)
    else: 
        group_l = list()
        group_m = list()
        indeces = list()
        for i in range(2,20):
            group_l.append(find_groups(w_mat,i))
            group_m.append(silhouette_score(w_mat,group_l[-1],metric='euclidean'))
            indeces.append(i)
        ideal = np.where(group_m == np.min(group_m))[0][0]

        pdb.set_trace()
        group_l = group_l[ideal] 
    return group_l 
def get_per_group_distance(x_val,group_labels): 
    n_labels = np.unique(group_labels)
    for e in n_labels: 
        simi = torch.zeros((x_val.shape[0],x_val.shape[0]),requires_grad=False)
        pdist = torch.nn.PairwiseDistance(p=2)
        sub_val = x_val[group_labels==e,:] 
        for i in range(x_val.shape[0]): 
            for j in range(x_val.shape[0]): 
                simi[i,j] = pdist(sub_val[i].reshape(1,-1),sub_val[j].reshape(1,-1))
        simi = simi.detach()
        simi_sum = torch.sum(s)
    pass 



def find_groups(w_mat,num_k):
    #given a matrix of weights. Cluster them using euclidean distance 
    # approach similar to what ablation study does 
    clust = KMeans(n_clusters=num_k)
    groups = clust.fit_predict(w_mat)
    return groups 


def get_weight_similarity(weights):
    #given the list of weights we had obtained calculate similarity matrix 
    # similarity metric is l2 distance of normalized weights 
    w_shape = weights.shape[0]
    simi = torch.zeros((w_shape,w_shape),requires_grad=False)
    pdist = torch.nn.PairwiseDistance(p=2)
    for i in range(w_shape):
        for j in range(w_shape): 
            simi[i,j]= pdist(weights[i].reshape(1,-1),weights[j].reshape(1,-1))
    simi = simi.detach()
    simi_sum = torch.sum(simi,dim=1).numpy().reshape(-1,1)
    return simi_sum 

def get_weight_2_drop(weights,percentage):
    import pdb 
    sim_scores = get_weight_similarity(weights) #get similarity measuree 
    indeces = sim_scores.argsort(axis=0) # sort similarity measure in increasing order.
    samples = np.hstack([sim_scores,indeces])
    sorted_sample = samples[np.argsort(samples[:,0],axis=0),:][::-1]
    percentage_interest = percentage
    num_samples =  int(samples.shape[0]*percentage_interest) # get relative amount to use 
    to_null = sorted_sample[0:num_samples,1].astype(np.int16) # get indices of top n corelated 
    #pdb.set_trace()
    return to_null # these will be the items set to zero 

def ablation_study(model,loader,mask_dict,mode='demo',groundTruthDf=None,model_name=None,weight_path=None,num_task_classes=None):
    layers_interest_names = find_conv_layers(model).keys()
    avg_contribs = list()
    for i,e in tqdm(enumerate(layers_interest_names),total=len(layers_interest_names)):
        model = reload_model(model_name,weight_path,num_task_classes,cuda_str="cuda:0")
        model.eval() 
        layers_interest = find_conv_layers(model)
        contribs = get_layer_contribs(model,layers_interest,i,loader,mask_dict,mode,groundTruthDf) 
        print(contribs)
        avg_contribs.append(contribs)
    return layers_interest.keys(),avg_contribs


def get_out_shape(self, input, output):
    #This is meant for the forwad hook 
    #records the output of a particular layer 
    self.out_dim = output.shape


def get_layer_contribs(model,layer_list,layer_idx,loader,mask_dict,mode,ground_truths,device='cuda'): 
    #mode should either be demo or task 
    """
    trying to adapt this debiasing code here 
    doing modificaiton where i  pass in the original ground truths file 
    """ 
    for i,e in  enumerate(layer_list.keys()): 
        if  i< layer_idx:
            continue
        layer_mask = mask_dict[e]
        for filt_k in layer_mask: 
            layer_list[e].weight[filt_k].copy_(torch.zeros_like( layer_list[e].weight[filt_k]))
    if mode=='task': 
        output = test_iterator(model,loader,'cuda')
    else: 
        output = test_bias_iterator(model,loader,'cuda')
    combined =  pd.merge(output,ground_truths,left_on='png_path',right_on='file')
    combined['malignancy_pred']=combined[['Task_0','Task_1']].values.argmax(axis=1)
    if mode=='demo':
        acc = calc_metrics(combined['Race'],combined['RacePred'])
    else: 
        acc = accuracy_score(combined['labels']==1,combined['malignancy_pred']==1)
    return acc 


def get_uni_contribs(layer_mask,attr):
    groups = np.unique(layer_mask.cpu())
    layer_mask = layer_mask.cpu()
    contribs = list()
    for e in groups:
        thigns = layer_mask[:,:,0,0]
        i_want = np.where(thigns==e)[1] # use one because you idx columns
        contrib = np.unique(attr[0,i_want,:,:])
        contribs.append(contrib)
    contribs = np.array(contribs)
    return np.mean(np.abs(contribs))
    

def calc_metrics(true_labels,preds):
    return accuracy_score(true_labels,preds)

def reload_model(model_name,weight_path,num_task_classes,cuda_str): 
    model = model_builder(model_name=model_name,
                        num_task_classes=num_task_classes,
                        model_weight=weight_path,num_layers=0).to(cuda_str) 
    for e in model.parameters(): 
        e.requires_grad = False 
    return model