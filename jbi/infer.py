import os 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
from .helpers.args import get_infer_args
import torch 
from .train import set_seeds 
from .helpers.transforms import gen_transforms
from .datasets.data_factory import get_dataset
from torch.utils.data import DataLoader
from .models.model_factory import model_loader
from .trainers.trainer_factory import load_trainer
set_seeds()

def infer_laoder(conf):
    batch_size = conf['trainer_args']["batch_size"]
    tr_transforms, ts_transforms = gen_transforms(conf)
    dl_dict = dict()
    ds_obj = get_dataset(conf)
    c_trx = ts_transforms
    ds_sub = ds_obj(conf=conf, split='test', transforms=c_trx, debug=conf["debug"])
    num_workers = conf['num_workers']
    dl_dict['infer'] = DataLoader(
        ds_sub,
        batch_size=batch_size,
        shuffle=False,
        persistent_workers=True,
        num_workers=num_workers,
    )
    return dl_dict



def main():
    conf = get_infer_args()
    ckpt_path = conf['model_weight']
    checkpoint = torch.load(ckpt_path)
    og_conf = checkpoint["conf"]
    og_conf['transform_conf'] = conf['transform_conf']
    og_conf['test_transforms'] = conf['test_transforms']
    og_conf['num_workers'] = conf['num_workers']
    og_conf["model_weight"] = ckpt_path
    og_conf['debug'] = conf['debug']
    dl_dict = infer_laoder(og_conf)
    print('hi')
    model = model_loader(og_conf)
    writer = None
    trainer_func = load_trainer(og_conf)
    trainer = trainer_func(model, writer, og_conf, dl_dict)
    trainer.load_best_model()
    for k, v in trainer.dls.items():
        ts_preds = trainer.test_model(v)
        ts_path = os.path.join(conf["output_dir"], f"{k}_preds.csv")
        ts_preds.to_csv(ts_path, index=False)

if __name__=='__main__':
    main()