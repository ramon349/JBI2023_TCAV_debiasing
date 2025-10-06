print("Importing Stuff")
from .train import get_loaders
from .models.model_factory import model_loader
from .trainers.trainer_factory import load_trainer
from .helpers.args import get_optuna_params
import numpy as np
import os
from copy import deepcopy

print("Doing the optuna imports")
from optuna import samplers
import optuna

print("Done the optuna imports")


class ParamSweeper:
    def __init__(self, config) -> None:
        self.conf = config
        optuna_log_dir = config["optuna_log"]
        self.conf["log_dir"] = optuna_log_dir
        log_dir = os.path.join(optuna_log_dir, "optuna_log.db")
        os.makedirs(optuna_log_dir, exist_ok=True)
        self.storage = f"sqlite:///{log_dir}"
        self.direction = config["direction"]

    def start(self):
        train_mode = self.conf["trainer"]
        match train_mode:
            case "adversarial":
                search_space = {"lambda": [0, 0.25, 0.5, 0.75, 1.0]}
            case "ErmTrainer":
                search_space = {
                    "learn_rate": [0.01, 0.0001, 0.001],
                    "batch_size": [64, 128, 256],
                }
            case _:
                search_space = None
        if search_space:
            my_sampler = samplers.GridSampler(search_space=search_space)
            self.n_trials: int = np.prod([len(v) for k, v in search_space.items()])
            study = optuna.create_study(
                storage=self.storage,
                study_name="my_study",
                load_if_exists=True,
                directions=self.direction,
                sampler=my_sampler,
            )
        else:
            self.n_trials: int = self.conf["n_trials"]
            study = optuna.create_study(
                storage=self.storage,
                study_name="my_study",
                load_if_exists=True,
                directions=self.direction,
            )
        print(f"The number of trials is {self.n_trials}")
        study.optimize(
            self.do_optim,
            n_trials=self.n_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )

    def do_optim(self, trial_obj):
        trainer_cls = load_trainer(self.conf)
        # call the class method to get trainer specific information
        ov_conf = deepcopy(self.conf)
        ov_conf = trainer_cls.get_trial_suggestions(trial_obj, ov_conf)
        model = model_loader(ov_conf)
        model_params = deepcopy(ov_conf["model_parameters"])
        model_params = model.get_trial_suggestions(trial_obj, model_params)
        # call the class method to get model specific optuna params
        # model = model_cls(model_params)
        model = model.to(ov_conf["device"][0])
        # reload the dataloader
        dls = get_loaders(self.conf)
        trainer = trainer_cls(
            model=model, conf=ov_conf, data_loaders=dls, tb_writter=None
        )
        return trainer._fit_optuna(trial_obj)


def main():
    print("Getitng The parameters")
    conf = get_optuna_params()
    print("initializing the param sweeper")
    sweeper = ParamSweeper(conf)
    print("about to start sweep")
    sweeper.start()


if __name__ == "__main__":
    main()
