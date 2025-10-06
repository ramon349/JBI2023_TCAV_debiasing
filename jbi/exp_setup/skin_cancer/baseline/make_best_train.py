from argparse import ArgumentParser
from ..skin_cancer_utils import _skin_transform_params
from pathlib import Path
import json
from typing import Union, Any
from .make_baseline_optim import _get_base_template
from ...exp_utils import _get_best_params


def _parse_args() -> dict[str, str]:
    args = ArgumentParser()
    args.add_argument("--csv_path", required=True, type=str)
    args.add_argument("--config_dir", required=True, type=str)
    args.add_argument("--log_dir", required=True, type=str)
    args.add_argument("--optuna_log", required=True, type=str)
    # args.add_argument("--train_mode", required=True, type=str)
    return vars(args.parse_args())


def main():
    conf = _parse_args()
    template = _get_base_template()
    log_dir = Path(conf["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    config_dir = Path(conf["config_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    csv_path = conf["csv_path"]
    template["csv_path"] = csv_path
    template["log_dir"] = str(log_dir)
    config_path = str(config_dir / "train.json")
    script_path = str(config_dir / "run_train.sh")
    best_params = _get_best_params(conf["optuna_log"])
    print(f"Got Best Params to be")
    print(best_params)
    for k, v in best_params.items():
        template["trainer_args"][k] = v
    # write the config file
    with open(config_path, "w") as f:
        json.dump(template, f, indent=1)
    # write the bash script to  run the optimization script
    with open(script_path, "w") as f:
        print(f"python3 -m jbi.train --config_path {config_path}", file=f)


if __name__ == "__main__":
    main()
