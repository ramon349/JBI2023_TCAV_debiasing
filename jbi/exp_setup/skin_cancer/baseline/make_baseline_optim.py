from argparse import ArgumentParser
from ..skin_cancer_utils import _skin_transform_params
from pathlib import Path
import json
from typing import Union, Any


def _get_base_template() -> dict[str, Any]:
    transform_info = _skin_transform_params()
    template = {
        "csv_path": None,
        "num_workers": 32,
        "device": ["cuda:0"],
        "batch_size": 128,
        "dataset": "ImageDataMask",
        "model": "densenet121",
        "trainer": "ErmTrainer",
        "col_info": {
            "img_col": "file",
            "task_col": "three_partition_label_cls",
            "mask_col": "mask_file",
        },
        "trainer_args": {
            "epochs": 100,
            "grad_step": 1,
            "learn_rate": 0.001,
            "early_stop": 20,
            "grad_norm": 0,
        },
        "model_parameters": {"num_task": 3},
        "splits": ["train", "test", "val"],
    }
    # Add the transform information
    for k, v in transform_info.items():
        template[k] = v
    return template


def _parse_args() -> dict[str, str]:
    args = ArgumentParser()
    args.add_argument("--csv_path", required=True, type=str)
    args.add_argument("--config_dir", required=True, type=str)
    args.add_argument("--optuna_log_dir", required=True, type=str)
    return vars(args.parse_args())


def main():
    conf = _parse_args()
    template = _get_base_template()
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
    # write the config file
    with open(config_path, "w") as f:
        json.dump(template, f, indent=1)
    # write the bash script to  run the optimization script
    with open(script_path, "w") as f:
        print(f"python3 -m jbi.optimize --config_path {config_path}", file=f)


if __name__ == "__main__":
    main()
