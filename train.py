import os
import sys
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from plain_model_dist import model_builder
from torch_transforms import *
from data_iterators.batch_iterators import classic_batch_iterator,debias_batch_iterator
from helper_funcs import count_parameters
from testCode import test_iterator,test_bias_iterator
from helper_funcs import build_transform,get_loss,seed_everything
    
seed_everything(1996)
torch.multiprocessing.set_sharing_strategy("file_system")

def run_train_loop(model,
            phase,
            Data_loader,
            optimizer,
            device,
            epoch,
            dset,
            writer,
            train_mode='single',
            config=None,
            global_step_count=None,
        ):
        if train_mode =='single':  
            #classig_train_loop
            return classic_batch_iterator(model,phase,Data_loader,optimizer,device,epoch,dset,writer=writer,config=config,global_step_count=global_step_count)
        if train_mode =='debias': 
            #debiasing_train_loop
            return debias_batch_iterator(model,phase,Data_loader,optimizer,device,epoch,dset,writer=writer,config=config,global_step_count=global_step_count)

from helper_funcs import figure_version,build_loaders,get_optimizer,get_scheduler
def main():
    print("Loading training config")
    config = json.load(open(sys.argv[1], "r"))
    # name of model to be loaded. defined in plain_model_dist.py 
    model_name = config["model_name"] 
    # if we are fine tuning  an existing model we use pass in the absolute path to the weights 
    weight_path = f"{config['weight_path']}"
    #shapes  images will be resized to during training 
    img_shape = (config["img_shape1"], config["img_shape2"])
    #gpus to use -1 if no gpu 
    cuda_num1 = int(config["cuda_num1"])
    cuda_num2 = int(config["cuda_num2"])
    #how many classes are predicting 
    task_classes = config["task_num_classes"]
    # used to create our log  files 
    data_name = config["data_name"] 
    train_mode = config['train_mode']
    exp_name = f"exp_{model_name}_{img_shape[0]}_{data_name}_{train_mode}"
    path = f"./logs/{model_name}/{exp_name}"
    version = figure_version(path) #assign each run a unique number so  /baseline_exp/run1 or /baseline_exp/run2
    #path of my log file 
    path = f"./logs/{model_name}/{exp_name}/{version}"
    epochs = config["epochs"]
    (best_epoch, best_loss) = (-1, 999999999)  # sentinel values for training
    # get my data loaders with corresponding transforms 
    (train_loader, test_loader, val_loader) = build_loaders(config)
    DEVICE = torch.device(f"cuda:{cuda_num1}" if torch.cuda.is_available() else "cpu")
    # Set up my model 
    model = model_builder(
        model_name,
        config=config
    )
    model=model.to(torch.float32).to(DEVICE) 
    #this is the debiasing lambda only used here for  logging validation loss
    (trainable_params, total_params) = count_parameters(model)
    print(f" Trainable params: {trainable_params}  total_params: {total_params}")
    # Get Multiple GPU set up #TODO:  Distributed Data Parallel approach with n=1 
    if cuda_num2 > -1:
        model = torch.nn.DataParallel(model, device_ids=[cuda_num1, cuda_num2])
    #model = torch.compile(model,dynamic=True,fullgraph=False)
    optimizer = get_optimizer(model, config=config)
    scheduler = get_scheduler(optimizer, config)
    writer = SummaryWriter(path)
    torch.cuda.empty_cache()
    ckpt_path = f"./model_checkpoints/exp_{model_name}_{img_shape[0]}_{data_name}/{version}/"
    f_name = f"{ckpt_path}model_w.ckpt"
    global_step_count = 0 
    print(config['train_mode']) 
    for epoch in range(epochs):
        train_d = run_train_loop(
            model=model,
            phase="train",
            Data_loader=train_loader,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch,
            writer=writer,
            dset="train",
            config=config,
            train_mode=config['train_mode'],
            global_step_count=global_step_count)
        global_step_count = train_d['global_step_count']
        optimizer.zero_grad()
        #run validation step 
        #note it says test because we are in testing mode 
        train_d = run_train_loop(
            model=model,
            phase="test",
            Data_loader=val_loader,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch,
            writer=writer,
            dset="val",
            train_mode=config['train_mode'],
            config=config
        ) 
        val_loss = train_d['ov_loss'] 
        if config['scheduler'] =='StepLR': 
            scheduler.step()
        else: 
            scheduler.step(val_loss)
        writer.add_scalar("val_loss",val_loss)
        if val_loss < best_loss:
            best_epoch = epoch
            best_loss = val_loss
            os.makedirs(ckpt_path, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss":val_loss, 
                    "config":config
                },
                f_name,
            )
            print(f"Saving epochs {epoch} obtained loss of {best_loss}")
        
        if (epoch - best_epoch) >= config['patience']:
            print("NO IMPROVEMENTS WEA RE DONE")
            break
    model.load_state_dict(torch.load(f_name)['model_state_dict'])
    os.makedirs('./results',exist_ok='true')
    if train_mode =='single':
        val_df = test_iterator(model,val_loader,DEVICE)
        test_df = test_iterator(model,test_loader,DEVICE)
    else: 
        val_df = test_bias_iterator(model,val_loader,DEVICE)
        test_df = test_bias_iterator(model,test_loader,DEVICE)
    val_df.to_csv('./results/'+exp_name+'_val_preds.csv',index=False) 
    test_df.to_csv('./results/'+exp_name+'_test_preds.csv',index=False) 

if __name__ == "__main__":
    main() 
