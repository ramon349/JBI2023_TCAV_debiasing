import os
import sys
import json
import torch
from glob import glob
from torch.utils.data.dataloader import DataLoader
from plain_model_dist import model_builder
from torch_transforms import *
from data_iterators.batch_iterators import test_iterator
from loader_factory import get_factory 
from helper_funcs import build_transform 
from helper_funcs import build_transform
from torch.nn import functional 
from plain_model_dist import remove_module
import pandas as pd 

torch.multiprocessing.set_sharing_strategy("file_system")
torch.autograd.set_detect_anomaly(True)
def build_loaders(config):
    transform, val_transform = build_transform(config)
    task_classes = config["task_num_classes"]
    loader_mode = config["loader_mode"]
    get_loader =  get_factory(config['loader'])
    dataset = get_loader(loader_mode)
    num_workers = config["num_workers"]
    batch_size = config["batch_size"]
    # make dataset
    hold_val_data = dataset(
        config["train_file"],
        transform=val_transform,
        num_task_classes=task_classes,
        split='test',
    )
    # make dataloaders
    val_loader = DataLoader(
        hold_val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return ( val_loader)
def test_iterator(model,  Data_loader, device):
    # --------------------  Initial paprameterd
    task_gts = []
    task_preds = []
    img_paths = []
    for i, data in enumerate(Data_loader):
        print(f" Batch {i} ", end="\r")
        imgs, task_labels ,img_path = data
        imgs = imgs.to(device)
        task_labels = task_labels.to(device)
        img_paths.extend(img_path)
        for label in task_labels.cpu().numpy().tolist():
            task_gts.append(label)
        model.eval()
        with torch.no_grad(): 
            model_preds =  model(imgs) 
            task_pred = model_preds
            for pred in functional.softmax(task_pred.cpu()).numpy().tolist():
                task_preds.append(pred)
    task_preds = np.vstack(task_preds)
    num_task = task_preds.shape[1]
    img_paths = np.vstack(img_paths)
    mega_mat = np.hstack((task_preds, img_paths))
    task_names = [f'Task_{e}' for e in range(num_task)] 
    col_names = task_names + ['png_path']
    rows = range(mega_mat.shape[0])
    print(len(col_names))
    mega_df = pd.DataFrame(
        mega_mat, columns=col_names 
    )
    return mega_df
def test_bias_iterator(model,  Data_loader, device):
    # --------------------  Initial paprameterd
    demo_gts = []
    task_gts = []
    demo_preds = []
    task_preds = []
    img_paths = []
    for i, data in enumerate(Data_loader):
        print(f" Batch {i} ", end="\r")
        imgs, task_labels , demo_labels,img_path = data
        imgs = imgs.to(device)
        task_labels = task_labels.to(device)
        demo_labels = demo_labels.to(device)
        img_paths.extend(img_path)
        for label in demo_labels.cpu().numpy().tolist():
            demo_gts.append(label)
        for label in task_labels.cpu().numpy().tolist():
            task_gts.append(label)
        model.eval()
        with torch.no_grad():
            task_pred, demo_pred = model(imgs)
            for pred in functional.softmax(task_pred.cpu()).numpy().tolist():
                task_preds.append(pred)
            for pred in functional.softmax(demo_pred.cpu()).numpy().tolist():
                demo_preds.append(pred)
    task_preds = np.vstack(task_preds)
    num_task = task_preds.shape[1]
    demo_preds = np.vstack(demo_preds)
    num_demo = demo_preds.shape[1]
    img_paths = np.vstack(img_paths)
    mega_mat = np.hstack((task_preds, demo_preds, img_paths))
    task_names = [f'Task_{e}' for e in range(num_task)] 
    demo_names = [f'demo_{e}' for e in range(num_demo)] 
    col_names = task_names + demo_names  + ['png_path']
    rows = range(mega_mat.shape[0])
    print(len(col_names))
    mega_df = pd.DataFrame(
        mega_mat, columns=col_names 
    )
    return mega_df
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



def main():
    print("Loading training config")
    with open( sys.argv[1],'rb' ) as f :  
        checkpoint = torch.load(f,map_location='cpu')
    # experiiment parameters
    # loading experiment information
    config = checkpoint['config']
    model_name = config["model_name"]
    weight_path = f"{config['weight_path']}"
    img_shape = (config["img_shape1"], config["img_shape2"])
    cuda_num1 = int(config["cuda_num1"])
    cuda_num2 = int(config["cuda_num2"])
    task_classes = config["task_num_classes"]
    if 'dem_num_classes' in config: 
        demo_classes = config["dem_num_classes"]
    data_name = config["data_name"]
    exp_name = f"exp_{model_name}_{img_shape[0]}_{data_name}"
    path = f"./logs/{model_name}/{exp_name}"
    version = figure_version(path)
    path = f"./logs/{model_name}/{exp_name}/{version}"
    # establish which model transforms i'll be using
    val_loader = build_loaders(config)
    DEVICE = torch.device(f"cuda:{cuda_num1}" if torch.cuda.is_available() else "cpu")
    # Set up my model
    if config['train_mode'] == 'debias':
        model = model_builder(model_name,
        num_task_classes=task_classes,
        num_layers=0,config=config)
    else: 
        model = model_builder(
            model_name,
            num_task_classes=task_classes,
            model_weight=weight_path, 
            num_layers=0
            ) 
    model.load_state_dict(remove_module(checkpoint['model_state_dict'],"module."))
    model = model.to(DEVICE)
    # Get Multiple GPU set up
    if cuda_num2 > -1:
        model = torch.nn.DataParallel(model, device_ids=[cuda_num1, cuda_num2])
    split =  'test'
    os.makedirs('./results',exist_ok='true')
    if config['train_mode']=='debias':
        test_df = test_bias_iterator(model,val_loader,DEVICE)
    else: 
        test_df = test_iterator(model,val_loader,DEVICE)
    test_df.to_csv('./results/'+exp_name+'_'+split+'_preds.csv',index=False) 


if __name__ == "__main__":
    main()
