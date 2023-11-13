import os
import sys
import json
import torch
from glob import glob
from plain_model_dist import model_builder,multi_task_model_builder
from torch_transforms import *
from loader_factory import get_factory 
from torch_transforms import AddGaussNoise
from helper_funcs import build_transform
import pandas as pd 
import numpy as np 
import os 
from memory_profiler import profile

torch.set_num_threads(4)


def build_loaders(config):
    transform, val_transform = build_transform(config)
    demo_classes = config["demo_num_classes"]
    task_classes = config["task_num_classes"]
    loader_mode = config["loader_mode"]
    get_loader =  get_factory(config['loader'])
    dataset = get_loader(loader_mode)
    num_workers = config["num_workers"]
    batch_size = 32 #config["batch_size"]
    prefecth = config["prefetch_factor"]
    # make dataset
    hold_val_data = dataset(
        config["train_file"],
        transform=val_transform,
        num_task_classes=task_classes,
        split=config['split_name'],
    )
    return hold_val_data
def embed_iterator(model,  dataset, device):
    # --------------------  Initial paprameterd
    task_preds = []
    img_paths = []
    t_labels = list() 
    labels = list()  
    print(dataset)
    with torch.inference_mode(mode=True):
        for i  in range(len(dataset)):
            data =  dataset.__getitem__(i)
            print(f" Batch {i} ", end="\r")
            imgs0,t_label,label,paths = data
            imgs0 = imgs0.to(device)
            imgs0_embed = model.embed(imgs0.unsqueeze(0))
            task_preds.append(imgs0_embed.cpu())
            img_paths.append(paths)
            labels.append(label)    
            t_labels.append(t_label)
    task_preds = np.vstack(task_preds)
    img_paths = np.hstack(img_paths)
    labels = np.hstack(labels) 
    t_labels = np.hstack(t_labels)
    return (task_preds,img_paths,labels,t_labels)

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


@profile 
def main():
    print("Loading training config")
    config = json.load(open(sys.argv[1], "r"))
    # experiiment parameters
    # loading experiment information
    model_name = config["model_name"]
    weight_path = f"{config['weight_path']}"
    img_shape = (config["img_shape1"], config["img_shape2"])
    cuda_num1 = int(config["cuda_num1"])
    cuda_num2 = int(config["cuda_num2"])
    demo_classes = config["demo_num_classes"]
    task_classes = config["task_num_classes"]
    data_name = config["data_name"]
    exp_name = f"exp_{model_name}_{img_shape[0]}_{data_name}"
    path = f"./logs/{model_name}/{exp_name}"
    version = figure_version(path)
    path = f"./logs/{model_name}/{exp_name}/{version}"
    epochs = config["epochs"]
    (best_epoch, best_loss) = (-1, 999999999)  # sentinel values for training
    # establish which model transforms i'll be using
    val_loader = build_loaders(config)
    #val_loader.data = val_loader.data.sample(100)
    DEVICE = torch.device(f"cuda:{cuda_num1}" if torch.cuda.is_available() else "cpu")
    # Set up my model
    if 'task' in config and config['task']=='debias':
        model= multi_task_model_builder(model_name,num_task_classes=task_classes,num_demo_classes=demo_classes,num_layers=0).to(DEVICE)
    else: 
        model = model_builder(
            model_name,
            num_task_classes=task_classes,
            model_weight=weight_path
        ).to(DEVICE)
    # Get Multiple GPU set up
    if cuda_num2 > -1:
        model = torch.nn.DataParallel(model, device_ids=[cuda_num1, cuda_num2])
    os.makedirs('./results',exist_ok='true')
    result_prefix = config['test_name']
    test_preds,paths,img_labels,t_labels =embed_iterator(model,val_loader,DEVICE)
    meta_df = pd.DataFrame({'paths':paths,'labels':img_labels,'severity':t_labels}) 
    np.savetxt(f'./results/{result_prefix}_embeds.txt',test_preds,delimiter='\t')
    meta_df.to_csv(f'./results/{result_prefix}_meta_df.csv',sep='\t')



if __name__ == "__main__":
    main()
