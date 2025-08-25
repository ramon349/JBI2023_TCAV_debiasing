from .helpers.args import get_train_args
from .datasets.data_factory import get_dataset
from .helpers.transforms import gen_transforms
from torch.utils.data import DataLoader
from .models.model_factory import model_loader
from .trainers.trainer_factory import load_trainer
from glob import glob
from torch.utils.tensorboard.writer import SummaryWriter
import os
from .train import get_loaders
import torch
import sys


def main():
    ckpt_path = sys.argv[1]
    checkpoint = torch.load(ckpt_path)
    conf = checkpoint["conf"]
    conf["mdoel_weight"] = ckpt_path
    conf["test_transforms"] = ["toTensor", "resize", "norm"]
    dl_dict = get_loaders(conf)
    model = model_loader(conf)
    writer = None
    trainer_func = load_trainer(conf)
    trainer = trainer_func(model, writer, conf, dl_dict)
    trainer.load_best_model()
    for k, v in trainer.dls.items():
        ts_preds = trainer.test_model(v)
        ts_path = os.path.join(conf["log_dir"], f"{k}_preds.csv")
        ts_preds.to_csv(ts_path, index=False)


if __name__ == "__main__":
    main()
