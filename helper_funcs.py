from metric_util import calculate_metrics
from torch.utils.data import WeightedRandomSampler
import torch 
from torchvision import transforms
from torch_transforms import myRep,AddGaussNoise,typeConv,toGPU,make8bit
from torch.utils.data import DataLoader 
from loader_factory import get_factory
from functools import partial
import torch.nn as nn  
import torch.nn.functional as F
from glob import glob 
import random 
import os 
import numpy as np 
from torch import optim 
from torch.optim import lr_scheduler
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def figure_version(path):
    #  when saving model  checkpoints and logs. Need to make sure i don't overwrite previous experiemtns
    avail = glob(f"{path}/version_*")
    if len(avail) == 0:
        ver = "version_0"
    else:
        avail = sorted(avail, key=lambda x: int(x.split("_")[-1]))
        oldest = int(avail[-1].split("_")[-1])
        ver = f"version_{oldest+1}"
    return ver


def get_optimizer(model, config):
    optimizer = None
    if config["optim"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    if config["optim"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config["lr"])
    if config["optim"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"],weight_decay=0.05)
    return optimizer


def get_scheduler(optimizer, config):
    if config["scheduler"] == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["factor"],
            patience=5,
            threshold=0.001,
            threshold_mode="rel",
        )
    if config["scheduler"] == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config["factor"])
    return scheduler
def build_loaders(config):
    transform, val_transform = build_transform(config)
    task_classes = config["task_num_classes"]
    loader_mode = config["loader_mode"]
    get_loader =  get_factory(config['loader'])
    dataset = get_loader(loader_mode)
    batch_size = config["batch_size"]
    num_workers = 32 
    # make dataset
    train_data = dataset(
        config["train_file"],
        transform=transform,
        num_task_classes=task_classes,
        split="train",
    )
    val_data = dataset(
        config["train_file"],
        transform=val_transform,
        num_task_classes=task_classes,
        split="val",
    )
    hold_val_data = dataset(
        config["train_file"],
        transform=val_transform,
        num_task_classes=task_classes,
        split="test",
    )
    # make weighted sampler
    if config['weightSample']: 
        shuffle = False
        train_sampler = makeWeightedSampler(
            train_data,config['weightCat']
        ) 
    else: 
        shuffle = True
        train_sampler = None 
    # train_sampler = None
    # make dataloaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=train_sampler,
    )
    test_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        hold_val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return (train_loader, test_loader, val_loader)


##### Functions related to model transforms ######
def make_img_8bit(img):
    return  ((torch.maximum(img,torch.tensor(0))/img.max() ) * 255).to(torch.uint8)
def build_transform(config):
    # these are the transforms used by my model
    train_transforms = [get_transform(e,config) for e in config['train_transforms']]
    val_transform = [get_transform(e,config) for e in config['test_transforms']]
    train_transforms = transforms.Compose(train_transforms)
    val_transform = transforms.Compose(val_transform)
    return train_transforms, val_transform

def get_transform(names,config): 
    """ Given the name of a transform we build said torch transform. Using params from config as needed 
    names: str specifying which transfrom to build 
    config: dict containing parameters used by all transforms  
    """
    if names=='norm': 
        mu = config['norm_mu']=config['norm_mu']
        std = config['norm_std']=config['norm_std']
        return transforms.Normalize(mu,std)
    if names =='resize':
        shape0 = config['img_shape1']
        shape1 = config['img_shape2']
        return transforms.Resize((shape0,shape1))
    if names =='horizontal':
        return transforms.RandomHorizontalFlip(p=0.5)
    if names =='vertical':
        return transforms.RandomVerticalFlip(p=0.5)
    if names=='affine':
        return transforms.RandomAffine(15)
    if names=='noise':
        return AddGaussNoise(p=0.5)
    if names =='rep': 
        return myRep() 
    if names =="tensor": 
        return transforms.ToTensor()
    if names == "typeConv":
        return typeConv() 
    if names=='make8bit':
        return make8bit()
    if names =='centerCrop':
        return transforms.CenterCrop((224,224))
    if names == 'ColorJitter':
        if 'brightness' in config: 
            bright = config['brightness']
        if 'contrast' in config:
            contrast = config['contrast']
        if 'saturation' in config: 
            saturation = config['saturation']
        return transforms.ColorJitter(brightness=(bright[0],bright[1]),contrast=contrast,saturation=saturation) 



def log_perfs(true_l, preds, num_classes, writer, phase, task, loss, epoch,class_names=None,config=None):
    """ writes several metrics from my model training procedure 
    """
    #return if the wirter object is none 
    if not writer: 
        return 
    preds = F.softmax(preds,dim=1) 
    if phase=='val':
        writer.add_histogram('val_probs_pos',preds[:,1],global_step=epoch)
        writer.add_histogram('val_probs_neg',preds[:,0],global_step=epoch)
    metrics = calculate_metrics(true_l, preds, num_classes,class_names=class_names)
    for e in metrics.keys():
        writer.add_scalar(f"{phase}_{task}_{e}", metrics[e], global_step=epoch)
       

def print_config(conf):
    # used to output the training config of my model
    for e in conf.keys():
        print(f" {e} : {conf[e]}")


def count_parameters(model):
    """ Counts number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params, total_params


def makeWeightedSampler(dataset, group):
    """makes a weighted sampler based on the input dataset"""
    # Compute class weight each class should get the same weight based on its balance
    temp_df = dataset.data
    classes = temp_df[group].unique()
    classes = sorted(classes)
    # get the count of each unique
    cls_counts = {  sub_group:(temp_df[group]== sub_group).sum() for sub_group in classes }
    cls_weights = {e: 1/cls_counts[e] for e in cls_counts.keys() }
    for e in cls_weights.keys(): 
        print(f" Class :{e} will have weight {cls_weights[e]}")
    # make a tensor of the weights iterate through each row 
    sample_weights = []
    for label in temp_df[group]:
       sample_weights.append(cls_weights[label]) 
    sample_weights = torch.tensor(sample_weights)
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def get_loss(loss_name,reduction='none'):
    """ Construct loss functions 
    """
    if loss_name =="NLL":
        return partial(loss_application,loss_func=nn.NLLLoss(reduce=reduction),active_func=nn.LogSoftmax(dim=1))
    if loss_name=='BCE':
        return partial(loss_application,loss_func=nn.BCELoss(reduce=reduction),active_func=nn.Sigmoid(dim=1))
    if loss_name =="CE":
        return partial(loss_application,loss_func=nn.CrossEntropyLoss(reduce=False,reduction=None),active_func=None)

def loss_application(x,y,loss_func=None,active_func=None):  
    """  apply a loss function 
    """
    if active_func:
        return loss_func(active_func(x),y)
    else: 
        return loss_func(x,y)
