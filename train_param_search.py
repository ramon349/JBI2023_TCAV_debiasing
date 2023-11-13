import sys
import json
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from glob import glob
from torch.utils.tensorboard import SummaryWriter
from plain_model_dist import model_builder
from torch_transforms import *
from helper_funcs import count_parameters , build_loaders,seed_everything,get_loss,figure_version
import optuna 
from optuna.trial import TrialState
from train import run_train_loop
import logging
import optuna 
import os 
import random 
torch.multiprocessing.set_sharing_strategy("file_system")
seed_everything(1996)
import pdb 
def get_optimizer(model, config,trial):
    optimizer = None 
    if 'optim' not in config:
        #if the optimizer option isn't specified we search for an ideal 
        options = config['optim_opts'] if 'optim_opts' in config else ['SGD','Adam','AdamW']
        config["optim"] = trial.suggest_categorical("optimizer", options )
        config["lr"] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    if config["optim"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    if config["optim"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config["lr"])
    if config["optim"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    return optimizer

def get_scheduler(optimizer, config,trial):
    if 'scheduler' not in  config: 
        config["scheduler"] = trial.suggest_categorical(
            "scheduler", ["ReduceLROnPlateau", "StepLR"]
        )
        if config["scheduler"] == "StepLR":
            config["factor"] = trial.suggest_int("lr_step", 5, 10, 2)
        if config["scheduler"] == "ReduceLROnPlateau":
            config["factor"] = trial.suggest_float("platau_factor", 0.01, 0.1, step=0.02)
            config['patience'] = trial.suggest_int('patience',2,10,step=2)
    if config["scheduler"] == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["factor"],
            patience=config['patience'],
            threshold=0.001,
            threshold_mode="rel",
        )
    if config["scheduler"] == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config["factor"])
    return scheduler
def return_on_fail(val="hello"): 
    def decorate(f):
        def applicator(*args, **kwargs):
            try:
                return f(*args,**kwargs)
            except RuntimeError as e:
                print('We had a runtime oopsie')
                print(e.__cause__)
                pdb.set_trace()
                raise optuna.exceptions.TrialPruned()       
        return applicator
    return decorate

def main(config:dict,trial):
    #dict are modified in place. so you need to makea copy or else multiple studies will overwrite each other
    exp_config =  config.copy()   
    print("Loading training config")
    task = config['task']
    model_name = exp_config["model_name"]
    #TODO: why is img_shape not used here 
    img_shape = (exp_config["img_shape1"], exp_config["img_shape2"])
    cuda_num1 = int(exp_config["cuda_num1"])
    cuda_num2 = int(exp_config["cuda_num2"]) 
    task_classes = exp_config["task_num_classes"]
    if task =='debias': 
        demo_classes = exp_config["demo_num_classes"]
    epochs = exp_config["epochs"] 
    (best_epoch, best_loss) = (-1, 999999999)  # sentinel values for training
    # establish which model transforms i'll be using 
    if "batch_size" not in exp_config: 
        exp_config["batch_size"] = trial.suggest_int("batch_size", 16, 64, step=4) 
    (train_loader, test_loader, val_loader) = build_loaders(exp_config)
    DEVICE = torch.device(f"cuda:{cuda_num1}" if torch.cuda.is_available() else "cpu")
    # Set up my model
    demo_criterion = None # set it to none for useful purposes 
    print(model_name)
    if task =="debias": 
        exp_config['lambda'] = trial.suggest_float('lambda',0.01,1)
    model = model_builder(
        model_name,
        config=  exp_config 
    ).to(DEVICE)
    (trainable_params, total_params) = count_parameters(model)
    print(f" Trainable params: {trainable_params}  total_params: {total_params}")
    # Get Multiple GPU set up
    if cuda_num2 > -1:
        model = torch.nn.DataParallel(model, device_ids=[cuda_num1, cuda_num2])
    optimizer = get_optimizer(model, config=exp_config,trial=trial)
    scheduler = get_scheduler(optimizer, config=exp_config,trial=trial)
    writer = None 
    torch.cuda.empty_cache()
    global_step_count = 0  
    for epoch in range(epochs):
        train_d = run_train_loop(
            model=model,
            phase='train',
            Data_loader=train_loader,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch,
            writer=writer,
            dset='train',
            train_mode=task,
            config = exp_config ,
            global_step_count=global_step_count
        )
        global_step_count = train_d['global_step_count']
        optimizer.zero_grad()
        #note it says test because we are in testing mode 
        train_d =  run_train_loop(
            model=model,
            phase='test',
            Data_loader=val_loader,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch,
            writer=writer,
            dset=task,
            train_mode=task,
            config = exp_config,
            global_step_count=global_step_count
        ) 
        val_loss = train_d['ov_loss']
        if type(scheduler) == optim.lr_scheduler.ReduceLROnPlateau: 
            scheduler.step(val_loss)
        else: 
            scheduler.step() 
        if task !='debias':
            trial.report(val_loss,epoch)
            if  trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        if (epoch - best_epoch) >= 10:
            print("NO IMPROVEMENTS WEA RE DONE")
            break 
    if task =='debias':
        return (train_d['task_loss'],train_d['adv_loss'])
    else: 
        return train_d['task_loss']

if __name__ == "__main__":
    print("Loading training config")
    # these config files are the way i organize my experiemtns for data loading
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    config = json.load(open(sys.argv[1], "r"))
    study_name = config["study_name"]  # usually model being explored 
    storage_name = "sqlite:///{}.db".format(study_name)
    if config['task'] =='debias': 
        #we want to minimize classification loss but maximize  demographic classification loss 
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            directions=["minimize","maximize"],
            load_if_exists=True,
        )
    else: 
        study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
    )
    objective = lambda x: main(config, x)
    unique_trials = config['num_trials']
    while unique_trials > len(set(str(t.params) for t in study.trials)):
        study.optimize(objective, n_trials=1, timeout=100)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    if config['task']=='debias': 
        trial = study.best_trials[0]
    else: 
        trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
