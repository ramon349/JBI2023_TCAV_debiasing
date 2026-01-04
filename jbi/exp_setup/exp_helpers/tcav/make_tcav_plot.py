from argparse import ArgumentParser
from ..skin_cancer_utils import _skin_transform_params
from pathlib import Path
import json
from typing import Union, Any
from ...exp_utils import get_argmuents,get_transforms


def _parse_args() -> dict[str, str]:
    args = ArgumentParser()
    args.add_argument("--csv_path", required=True, type=str)
    args.add_argument("--config_dir", required=True, type=str)
    args.add_argument("--log_dir", required=True, type=str)
    args.add_argument("--weight_path", required=True, type=str)
    args.add_argument("--task",required=True,type=str,choices=['skin','mammo'])
    return vars(args.parse_args())


def _get_template(conf):
    data_info = get_argmuents(conf['task'])
    mask_col =  data_info.get("mask_col")
    template = {
        "csv_path": None,
        "col_info": {"img_col": data_info['img_col'], "task_col": data_info['task_col'],"mask_col":mask_col,'demo_col':data_info['demo_col']},
        "model_parameters": {"num_task": data_info['num_task'], "num_demo": data_info['num_demo']},
        "model": "DensenetTwoTask",
        "model_weight": None,
        "log_dir": None,
        "device": ["cuda:0"],
        "dataset": "single",
    }
    transform_info = get_transforms(conf['task']) 
    for k, v in transform_info.items():
        template[k] = v
    del template["train_transforms"]
    tcav_args = {} 
    tcav_args['split_col']= data_info['demo_col'] 
    tcav_args['group_a'] = data_info['group_a']
    tcav_args['group_b'] = data_info['group_b'] #group b is the ref group i.e majority
    tcav_args['samples_per_concept'] = 20
    template['transform_conf']["img_reader"] =data_info['img_reader']
    template['tcav_args'] = tcav_args
    return template


def main():
    conf = _parse_args()
    template = _get_template(conf=conf)
    log_dir = Path(conf["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    config_dir = Path(conf["config_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    csv_path = conf["csv_path"]
    template["csv_path"] = csv_path
    template["log_dir"] = str(log_dir)
    template["model_weight"] = conf["weight_path"]
    config_path = str(config_dir / "tcav.json")
    script_path = str(config_dir / "run_exp.sh")
    print(f"Got Best Params to be")
    # write the config file
    with open(config_path, "w") as f:
        json.dump(template, f, indent=1)
    # write the bash script to  run the optimization script
    with open(script_path, "w") as f:
        print(f"python3 -m jbi.tcav --config_path {config_path}", file=f)
    print(script_path)


if __name__ == "__main__":
    main()
