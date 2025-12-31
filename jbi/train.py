import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
from .helpers.args import get_train_args
from .datasets.data_factory import get_dataset
from .helpers.transforms import gen_transforms
from torch.utils.data import DataLoader
from .models.model_factory import model_loader
from .trainers.trainer_factory import load_trainer
from glob import glob
from torch.utils.tensorboard.writer import SummaryWriter
import torch 

def set_seeds():
 
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_seeds()
def get_loaders(conf):
    batch_size = conf['trainer_args']["batch_size"]
    tr_transforms, ts_transforms = gen_transforms(conf)
    dl_dict = dict()
    ds_obj = get_dataset(conf)
    num_workers = conf["num_workers"]
    for e in conf["splits"]:
        if e == "train":
            c_trx = tr_transforms
        else:
            c_trx = ts_transforms
        ds_sub = ds_obj(conf=conf, split=e, transforms=c_trx, debug=conf["debug"])
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


def make_writer(conf):
    log_dir = conf["log_dir"]
    log_files = glob(os.path.join(log_dir, "events*"))
    for e in log_files:
        os.remove(e)
    tb_writer = SummaryWriter(log_dir=log_dir)
    return tb_writer


def main():
    conf = get_train_args()
    dl_dict = get_loaders(conf)
    model = model_loader(conf)
    writer = make_writer(conf)
    trainer_func = load_trainer(conf)
    trainer = trainer_func(model, writer, conf, dl_dict)
    trainer.fit()
    trainer.load_best_model()
    for k, v in trainer.dls.items():
        ts_preds = trainer.test_model(v)
        ts_path = os.path.join(conf["log_dir"], f"{k}_preds.csv")
        ts_preds.to_csv(ts_path, index=False)


if __name__ == "__main__":
    main()
