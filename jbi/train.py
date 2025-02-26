from .helpers.args import get_train_args
from .datasets.data_factory import get_dataset
from .helpers.transforms import gen_transforms
from torch.utils.data import DataLoader
from .models.model_factory import model_loader
from .trainers.trainer_factory import load_trainer
from glob import glob
from torch.utils.tensorboard.writer import SummaryWriter
import os


def get_loaders(conf) ->dict :
    """  Uss config file to make DataLoader objects 
        conf:  experiment config file specifies experiment parameters 
        returns: 
            dictionary of dataloaders 
    """
    batch_size = conf["batch_size"]
    tr_transforms, ts_transforms = gen_transforms(conf)
    dl_dict = dict()
    #Loads the dataset Class 
    ds_obj = get_dataset(conf)
    num_workers = conf["num_workers"] 
    for e in conf["splits"]:
        if e == "train":
            c_trx = tr_transforms
        else:
            c_trx = ts_transforms
        #Initialize a dataset class with transforms based on split 
        ds_sub = ds_obj(conf=conf, split=e, transforms=c_trx) 
        #If debug parameter is passed we simply downsample the dataset 
        if 'debug' in conf and conf['debug']==True: 
                old_l = len(ds_sub)
                ds_sub.data = ds_sub.data.sample(frac=0.01,random_state=1996)
                new_l = len(ds_sub) 
                print(f"The {e} dset went from {old_l} down to {new_l}")
        sampler = None
        shuffle = True 
        dl_dict[e] = DataLoader(
            ds_sub,
            batch_size=batch_size,
            shuffle=shuffle,
            persistent_workers=True,
            num_workers=num_workers,
            sampler=sampler,
        )
    return dl_dict

def make_writer(conf) -> SummaryWriter:
    """ Conf will specify a directory of where to store tensorboard logs  

        returns: 
            Tensorboard SummaryWritter
    """
    log_dir = conf["log_dir"]
    log_files = glob(os.path.join(log_dir, "events*"))
    for e in log_files:
        os.remove(e)
    tb_writer = SummaryWriter(log_dir=log_dir)
    return tb_writer


def main():
    #Use argparse to obtain training parameters from conf file 
    conf = get_train_args() 
    #load the dataLoaderDictionary 
    dl_dict = get_loaders(conf)
    #Initialize the model of interest
    model = model_loader(conf)
    #Get our Tensorboard SummaryWritter
    writer = make_writer(conf)
    trainer_func = load_trainer(conf)
    #Initialize our Trainer Objer
    trainer = trainer_func(model, writer, conf, dl_dict)
    #Run the actual training code 
    trainer.fit() 
    #Once training is done load the best model and run testing 
    trainer.load_best_model() 
    for k,v in trainer.dls.items(): 
        ts_preds = trainer.test_model(v) 
        ts_path = os.path.join(conf['log_dir'],f'{k}_preds.csv')
        ts_preds.to_csv(ts_path,index=False)


if __name__ == "__main__":
    main()