import torch
from helper_funcs import *
from torch.nn import functional
import pandas as pd 
import numpy as np

def gen_weight_tensor(cls_weights,ground_truth):
    weights = torch.zeros(ground_truth.shape)
    #iterate through the columns 
    for j in range(ground_truth.shape[1]):
        #setting the sample weights for each finding 
        for i in range(ground_truth.shape[0]):
            cls_w_dict = cls_weights[j]
            if ground_truth[i,j]==0:
                weights[i,j] =  cls_w_dict['0.0']
            else: 
                ground_truth[i,j]= cls_w_dict['1.0']
    return weights

def classic_batch_iterator(
    model,
    phase,
    Data_loader,
    optimizer,
    device,
    epoch,
    dset,
    writer=None,
    early_stop =False, 
    stop_batch = 1500,
    config=None,
    global_step_count=0
):
    task_criterion = get_loss(config['task_loss'])
    running_loss = 0.0
    task_gts = []
    task_preds = [] 
    for i, data in enumerate(Data_loader): 
        if early_stop and i> stop_batch: 
            break 
        imgs, task_labels, _ = data
        if i == 0 and writer:
            #Log  the first batch of every epoch into tensorboard 
            #I use the make_img_8bit function so imgs are stored passed as 8bit (hXwx3) format. Otherwise blanks 
            writer.add_images(f"{dset}_{phase}_img", make_img_8bit(imgs.cpu().detach()), epoch)
        batch_size = imgs.shape[0]
        imgs = imgs.to(device)
        task_labels = task_labels
        if phase == "train":
            optimizer.zero_grad()
            model.train()
            task_pred  = model(imgs)
            task_pred =  (task_pred.cpu()).type(torch.float)
            task_loss = task_criterion(task_pred, task_labels).mean()
            task_loss.backward()
            optimizer.step() 
            running_loss += task_loss.detach() * batch_size
            if writer: 
                writer.add_scalar(f"train_batch_loss",task_loss,global_step=global_step_count)
            global_step_count +=1 
        else:
            for label in task_labels.cpu():
                task_gts.append(label)
            model.eval()
            with torch.no_grad():
                task_pred = model(imgs)
                for pred in task_pred.cpu():
                    task_preds.append(pred)
    if phase == "test":
        task_preds = torch.stack(task_preds)
        task_gts = torch.stack(task_gts)
        task_loss = task_criterion(task_preds.type(torch.float), task_gts).mean() 
        print(" Testing Stuff", end="\r")
        log_perfs(
            task_gts,
            task_preds,
            Data_loader.dataset.num_task_classes,
            writer,
            dset,
            'task',
            task_loss,
            epoch,
            class_names=Data_loader.dataset.class_names,
            config=config
        )
    return {'ov_loss':task_loss,'task_loss':task_loss,'global_step_count':global_step_count}

def switch_bias_mode(model,mode_val): 
    """ Modify the config
    """
    if torch.nn.parallel.DataParallel == type(model):
        model.module.set_reversal(mode_val)
    else:
        model.set_reversal(mode_val)
    
def debias_batch_iterator(
    model,
    phase,
    Data_loader,
    optimizer,
    device,
    epoch,
    dset,
    writer=None,
    config=None,
    global_step_count=0
):
    task_criterion = get_loss(config['task_loss'])
    adv_criterion = get_loss(config['adv_loss'])
    running_loss = 0.0
    task_gts = []
    dem_gts = []
    task_preds = [] 
    dem_preds = []
    lambd = config['lambda']
    for i, data in enumerate(Data_loader):
        imgs, task_labels, dem_labels,_ = data
        if i == 0 and writer:
            writer.add_images(f"{dset}_{phase}_img", make_img_8bit(imgs.cpu().detach()), epoch)
        imgs = imgs.to(device)
        task_labels = task_labels
        if phase == "train":
            #Training steps 
            switch_bias_mode(model,False)
            optimizer.zero_grad()
            model.train()
            #first forward pass  
            task_pred,dem_pred  = model(imgs)
            task_pred =  (task_pred.cpu()).type(torch.float)
            dem_pred =  (dem_pred.cpu()).type(torch.float)
            task_loss = task_criterion(task_pred, task_labels) 
            adv_loss = adv_criterion(dem_pred,dem_labels) 
            ov_loss = torch.mean(adv_loss + adv_loss * lambd)
            ov_loss.backward() 
            optimizer.step()  # update weights
            #Do the second forward pass with the gradient reversal 
            switch_bias_mode(model,True)
            task_pred,dem_pred  = model(imgs)
            task_pred =  (task_pred.cpu()).type(torch.float)
            dem_pred =  (dem_pred.cpu()).type(torch.float)
            adv_loss = adv_criterion(dem_pred,dem_labels).mean()
            adv_loss.backward() 
            optimizer.step()
            if writer: 
                writer.add_scalar(f"train_batch_task_loss",task_loss.detach().mean(),global_step=global_step_count)
                writer.add_scalar(f"train_batch_adv_loss",adv_loss.detach().mean(),global_step=global_step_count)
            global_step_count +=1 
        else:
            #record predictions if not training 
            for label in task_labels.cpu():
                task_gts.append(label)
            for label in dem_labels.cpu():
                dem_gts.append(label)
            model.eval()
            with torch.no_grad():
                task_pred,dem_pred = model(imgs)
                for pred in task_pred.cpu():
                    task_preds.append(pred)
                for pred in dem_pred.cpu():
                    dem_preds.append(pred)
    if phase == "test":
        task_preds = torch.stack(task_preds)
        task_gts = torch.stack(task_gts)  
        dem_preds = torch.stack(dem_preds)
        dem_gts = torch.stack(dem_gts)  
        task_loss = task_criterion(task_preds.type(torch.float), task_gts)
        adv_loss = adv_criterion(dem_preds.type(torch.float), dem_gts)
        ov_loss = torch.mean(task_loss + adv_loss *lambd) 
        log_perfs(
            task_gts,
            task_preds,
            Data_loader.dataset.num_task_classes,
            writer,
            dset,
            'task',
            ov_loss,
            epoch,
            class_names=Data_loader.dataset.class_names,
            config=config
        )
        log_perfs(
            dem_gts,
            dem_preds,
            Data_loader.dataset.num_dem_classes,
            writer,
            dset,
            'demo',
            ov_loss,
            epoch,
            class_names=Data_loader.dataset.dem_class_names,
            config=config
        )
    return {'ov_loss':ov_loss,'task_loss':task_loss.detach().mean(),'adv_loss':adv_loss.detach().mean(),'global_step_count':global_step_count}

def test_iterator(model,  Data_loader, device):
    task_gts = []
    task_preds = []
    img_paths = []
    for i, data in enumerate(Data_loader):
        imgs, task_labels,img_path = data
        imgs = imgs.to(device)
        task_labels = task_labels.to(device)
        img_paths.extend(img_path)
        for label in task_labels.cpu().numpy().tolist():
            task_gts.append(label)
        model.eval()
        with torch.no_grad():
            task_pred= model(imgs)
            for pred in functional.softmax(task_pred.cpu()).numpy().tolist():
                task_preds.append(pred)
    task_preds = np.vstack(task_preds)
    num_task = task_preds.shape[1]
    img_paths = np.vstack(img_paths)
    mega_mat = np.hstack((task_preds, img_paths))
    task_names = [f'Task_{e}' for e in range(num_task)] 
    col_names = task_names  + ['png_path']
    mega_df = pd.DataFrame(
        mega_mat, columns=col_names 
    )
    return mega_df 