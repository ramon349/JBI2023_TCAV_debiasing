from argparse import ArgumentParser
from ..skin_cancer_utils import _skin_transform_params
from pathlib import Path
import json
from typing import Union, Any


def _parse_args() -> dict[str, str]:
    args = ArgumentParser()
    args.add_argument("--csv_path", required=True, type=str)
    args.add_argument("--config_dir", required=True, type=str)
    args.add_argument("--log_dir", required=True, type=str)
    args.add_argument("--weight_path", required=True, type=str)
    return vars(args.parse_args())


def _get_template():
    template = {
        "csv_path": None,
        "col_info": {"img_col": "file", "task_col": "three_partition_label_cls","mask_col": "mask_file",'demo_col':'discrete_fitz'},
        "tcav_args": {"samples_per_concept": 20},
        "model_parameters": {"num_task": 2, "num_demo": 3},
        "model": "DensenetTwoTask",
        "model_weight": None,
        "log_dir": None,
        "device": ["cuda:0"],
        "dataset": "single",
    }
    transform_info = _skin_transform_params()
    for k, v in transform_info.items():
        template[k] = v
    del template["train_transforms"]
    return template


def main():
    conf = _parse_args()
    template = _get_template()
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


if __name__ == "__main__":
    main()
