from .ErmTrainer import TwoTaskTrainer
from .trainer_factory import TrainerRegister
from tqdm import tqdm
import torch
import optuna


@TrainerRegister.register(cls_name="AdversarialTrainerVanilla")
class AdvTrainer(TwoTaskTrainer):
    def __init__(self, model, tb_writter, conf, data_loaders):
        super().__init__(model, tb_writter, conf, data_loaders)
        self.lmbd = self.trainer_args["lambda"]
        self.epoch_delay = self.trainer_args["adv_delay"]

    def build_criteria(self):
        # build the basic criteria. i.e classification ones
        super().build_criteria()

    def train_epoch(self):
        self.model.train()
        grad_step = self.grad_step
        for i, batch in tqdm(
            enumerate(self.dls["train"]), total=len(self.dls["train"])
        ):
            self.opti.zero_grad()
            img_in = batch[self.img_col]
            task = batch[self.task_col]
            demo = batch[self.demo_col]
            task = task.to(self.device)
            demo = demo.to(self.device)
            # phase 1
            task_h, demo_h = self.model(img_in.to(self.device), task="phase1")
            task_loss = self.criterions["task"](task_h, task)
            demo_loss = self.criterions["demo"](demo_h, demo)
            loss = task_loss + demo_loss
            loss.backward()
            # phase2
            task_h, demo_h = self.model(img_in.to(self.device), task="phase2")
            task_loss = self.criterions["task"](task_h, task)
            demo_loss = self.criterions["demo"](demo_h, demo)
            if self.epoch_delay <= self.c_epoch:
                loss = task_loss + self.lmbd * demo_loss
            else:
                loss = task_loss
            loss.backward()
            if (
                i % grad_step == 0
            ):  # do an update every two steps instead of every to accum
                self.opti.step()
            self._log_scalar("batch_ov_loss", loss, global_step=self.gb_step)
            self._log_scalar("batch_task_loss", task_loss, global_step=self.gb_step)
            self._log_scalar("batch_demo_loss", demo_loss, global_step=self.gb_step)
            self.gb_step += 1
        self.c_epoch += 1

    @staticmethod
    def get_trial_suggestions(cls, trial: optuna.Trial, c_conf):
        lmbd = trial.suggest_categorical("lambda", choices=[0, 0.25, 0.5, 0.75, 1.0])
        c_conf["trainer_args"]["lambda"] = lmbd
        return c_conf


@TrainerRegister.register("TCAVDebias")
class AdversarialDebiasTCAV(AdvTrainer):
    def __init__(self, model, tb_writter, conf, data_loaders):
        self.ablation_layer = conf["trainer_args"]["layer_debias"]
        self._freeze_weights(model, conf["trainer_args"]["layer_debias"])
        super().__init__(model, tb_writter, conf, data_loaders)

    def _freeze_weights(self, model, ablation_layer):
        """Given the name of a weight parameter we freeze all layers until we find it"""
        # get a list of all the parameters
        if ablation_layer:
            found = False
            names = [n for n, e in model.named_parameters()]
            idx = self.find_layer(ablation_layer, all_layer_names=names)
            for i, e in enumerate(model.parameters()):
                if i < idx:
                    e.requires_grad = False

    def find_layer(self, layer_name, all_layer_names):
        for i, e in enumerate(all_layer_names):
            if e.startswith(layer_name):
                return i
        raise ValueError(f"{layer_name} was not found")

    def train_epoch(self):
        self.model.train()
        grad_step = self.grad_step
        for i, batch in tqdm(
            enumerate(self.dls["train"]), total=len(self.dls["train"])
        ):
            self.opti.zero_grad()
            img_in = batch[self.img_col]
            task = batch[self.task_col].to(self.device)
            demo = batch[self.demo_col].to(self.device)
            # phase 1
            task_h, demo_h = self.model(img_in.to(self.device), task="phase1")
            task_loss = self.criterions["task"](task_h, task)
            demo_loss = self.criterions["demo"](demo_h, demo)
            loss = task_loss + demo_loss
            loss.backward()
            self.opti.step()
            # phase2
            task_h, demo_h = self.model(img_in.to(self.device), task="phase2")
            task_loss = self.criterions["task"](task_h, task)
            demo_loss = self.criterions["demo"](demo_h, demo)
            if self.epoch_delay <= self.c_epoch:
                loss = task_loss + self.lmbd * demo_loss
            else:
                loss = task_loss
            loss.backward()
            self.opti.step()
            self._log_scalar("batch_ov_loss", loss, global_step=self.gb_step)
            self._log_scalar("batch_task_loss", task_loss, global_step=self.gb_step)
            self._log_scalar("batch_demo_loss", demo_loss, global_step=self.gb_step)
            self.gb_step += 1
        self.c_epoch += 1
        self.model.eval()
