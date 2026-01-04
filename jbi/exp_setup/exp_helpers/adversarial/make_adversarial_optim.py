from argparse import ArgumentParser
from ..skin_cancer_utils import _skin_transform_params
from pathlib import Path
import json
from typing import Union, Any
from ...exp_utils import _get_best_params, get_transforms,get_argmuents


def _get_adv_template(conf) -> dict[str, Any]:
    transform_info = get_transforms(conf['task']) 
    data_info =  get_argmuents(conf['task'])
    layer_debias = conf['adv_layer']
    base_weight = conf['base_weight']
    task = conf['task']
    template = {
        "csv_path": None,
        "num_workers": 32,
        "device": ["cuda:0"],
        "dataset": 'TwoTaskMask' if task =='skin' else 'TwoTask', 
        "model": "DensenetTwoTaskAdv",
        "trainer": "TCAVDebias",
        "col_info": {
            "img_col":  data_info['img_col'],
            "task_col": data_info['task_col'],
            "demo_col":data_info['demo_col'], #TODO: Add demo col here
            "mask_col":  data_info.get("mask_col",""),
        },
        "trainer_args": {
            "epochs": 100,
            "grad_step": 1,
            "learn_rate": 0.001,
            "lambda": 0.5,
            "early_stop":20,
            "layer_debias": layer_debias,
            "grad_norm":0,
            "adv_delay":0,
        },
        "model_parameters": {"num_task": data_info['num_task'], "num_demo":data_info['num_demo'],'base_weight':base_weight},
        "splits": ["train", "test", "val"],
    }
    # Add the transform information
    for k, v in transform_info.items():
        template[k] = v
    template['transform_conf']["img_reader"] =data_info['img_reader']
    return template


def _parse_args() -> dict[str, str]:
    args = ArgumentParser()
    args.add_argument("--csv_path", required=True, type=str)
    args.add_argument("--config_dir", required=True, type=str)
    args.add_argument("--optuna_log_dir", required=True, type=str)
    args.add_argument("--base_config",required=True,type=str)
    args.add_argument("--adv_layer",required=True,type=str)
    args.add_argument("--base_weight",required=True,type=str)
    args.add_argument("--task",required=True,type=str,choices=['mammo','skin'])
    return vars(args.parse_args())

def copy_params(conf,new_confg): 
    with open(conf['base_config'],'r') as f: 
        old_conf = json.load(f)
    new_confg['trainer_args']['learn_rate']=old_conf['trainer_args']['learn_rate']
    new_confg['trainer_args']['batch_size']=old_conf['trainer_args']['batch_size']


def main():
    conf = _parse_args()
    template = _get_adv_template(conf)
    log_dir = Path(conf["optuna_log_dir"])
    config_dir = Path(conf["config_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    csv_path = conf["csv_path"]
    template["csv_path"] = csv_path
    template["optuna_log"] = str(log_dir)
    config_path = str(config_dir / "optimzie.json")
    script_path = str(config_dir / "run_optim.sh")
    template["direction"] = ["minimize"]
    template["n_trials"] = 20
    copy_params(conf=conf,new_confg=template)
    # write the config file
    with open(config_path, "w") as f:
        json.dump(template, f, indent=1)
    # write the bash script to  run the optimization script
    with open(script_path, "w") as f:
        print(f"python3 -m jbi.optimize --config_path {config_path}", file=f)


if __name__ == "__main__":
    main()
