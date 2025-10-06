from .trainer_factory import TrainerRegister
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from typing import Dict
from torch import optim
import torch
from tqdm import tqdm
from collections import defaultdict
from .trainer_factory import TrainerRegister
import os
from ..models.model_factory import remove_module
from ..helpers.helpers import proc_inference
from sklearn.metrics import balanced_accuracy_score
import optuna
import monai
from monai.data import meta_tensor
import pdb


@TrainerRegister.register(cls_name="ErmTrainer")
class BasicTrainer(object):
    def __init__(
        self,
        model: nn.Module,
        tb_writter: SummaryWriter,
        conf: Dict,
        data_loaders: Dict,
    ) -> None:
        self.model = model
        self.tb: SummaryWriter = tb_writter
        self.conf = conf
        self.dls = data_loaders
        self.trainer_args = conf["trainer_args"]
        self.total_epochs = self.trainer_args["epochs"]
        device = conf["device"][0]
        self.c_epoch = 0
        self.device = device
        self.gb_step = 0
        self.init_optims()
        self.best_tr_loss = 100
        self.save_interval = 1500
        self.build_criteria()
        self.grad_step = self.trainer_args["grad_step"]
        self.val_eval_step = 1
        self.early_stop = 5
        self._suppres_tqdm = False
        self.compile_model()
        self.early_stop = self.trainer_args["early_stop"]
        col_info = self.conf["col_info"]
        self.img_col = col_info["img_col"]
        self.task_col = col_info["task_col"]
        self.clip_grad = self.trainer_args["grad_norm"]

    def compile_model(self):
        pass

    def fit(self):
        num_epochs = self.total_epochs
        best_val_loss = 90000
        best_epoch = -1
        for i in range(num_epochs):
            self.train_epoch()
            self._log_scalar(
                "learning_rate", self.sch.get_last_lr()[0], global_step=self.c_epoch
            )
            if (i % self.val_eval_step) != 0:
                continue
            val_loss = self.val_epoch()
            self.sch.step(val_loss)
            if val_loss <= best_val_loss:
                self.store_model()
                best_val_loss = val_loss
                best_epoch = i
            if (i - best_epoch) >= self.early_stop:
                print("Going to do early breaking. 5 Epochs No Progress")
                break

    def _log_scalar(self, val_name, val, global_step=None):
        if self.tb:
            self.tb.add_scalar(val_name, val, global_step=global_step)

    def store_model(self):
        model_dir = self.conf["log_dir"]
        w_path = os.path.join(model_dir, "model_w.ckpt")
        torch.save(
            {
                "conf": self.conf,
                "model_weights": remove_module(self.model.state_dict()),
                "epoch": self.c_epoch,
            },
            f=w_path,
        )

    def load_best_model(self):
        model_dir = self.conf["log_dir"]
        w_path = os.path.join(model_dir, "model_w.ckpt")
        model_w = torch.load(w_path, map_location=self.device)
        self.model.load_state_dict(model_w["model_weights"])

    def init_optims(self):
        learn_rate = self.trainer_args["learn_rate"]
        name_list = list()
        for n, e in self.model.named_parameters():
            if e.requires_grad:
                name_list.append(n)
        print("Parameters to be updated are")
        print(name_list)

        self.opti = optim.AdamW(
            [e for e in self.model.parameters() if e.requires_grad], lr=learn_rate
        )
        self.sch = optim.lr_scheduler.ReduceLROnPlateau(
            self.opti, mode="min", patience=3
        )

    def build_criteria(self):
        self.criterions = dict()
        get_incidence = ("weight_task" in self.conf) and self.conf["weight_task"]
        if get_incidence:
            print("We are doing the task weighting")
            ws = self.dls["train"].dataset.get_inverse_weight("task")
            weight = torch.tensor(ws, dtype=torch.float)
        else:
            weight = None
        self.criterions["task"] = nn.CrossEntropyLoss(weight=weight)

    def train_epoch(self):
        self.model.train()
        grad_step = self.grad_step
        for i, batch in tqdm(
            enumerate(self.dls["train"]), total=len(self.dls["train"])
        ):
            self.opti.zero_grad()
            img_in = batch[self.img_col]
            task = batch[self.task_col]
            task_h = self.model(img_in.to(self.device)).cpu()
            loss = self.criterions["task"](task_h, task)
            loss.backward()
            if self.clip_grad:
                breakpoint()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            if (
                i % grad_step
            ) == 0:  # do an update every two steps instead of every to accum
                self.opti.step()
            self._log_scalar("batch_task_loss", loss, global_step=self.gb_step)
            self.gb_step += 1
        self.c_epoch += 1

    def val_epoch(self):
        self.model.eval()
        task_loss = 0
        with torch.no_grad():
            all_loss = 0
            val_len = len(self.dls["val"])
            for i, batch in enumerate(self.dls["val"]):
                img_in = batch[self.img_col]
                task = batch[self.task_col]
                task_h = self.model(img_in.to(self.device)).cpu()
                task_loss += self.criterions["task"](task_h, task).item()
            task_loss /= val_len
            self._log_scalar("val_task_loss", task_loss, global_step=self.c_epoch)
        return task_loss

    def _set_supress_tqdm(self, val):
        self._suppres_tqdm = val

    def test_model(self, dl):
        self.model.eval()
        with torch.no_grad():
            ground_truths = defaultdict(list)
            preds = defaultdict(list)
            all_paths = list()
            for i, batch in tqdm(
                enumerate(dl), total=len(dl), disable=self._suppres_tqdm
            ):
                img_in = batch[self.img_col]
                task = batch[self.task_col]
                img_path = batch[f"{self.img_col}_meta_dict"]["filename_or_obj"]
                task_h = self.model(img_in.to(self.device)).cpu()

                truth_names = ["task_t"]
                truth_vals = [task]
                pred_names = ["task_p"]
                pred_vals = [task_h]
                for n, e in zip(truth_names, truth_vals):
                    sub = tensor_convert(e)
                    ground_truths[n].extend(sub)
                for n, e in zip(pred_names, pred_vals):
                    sub = tensor_convert(e)
                    preds[n].extend(sub)
                all_paths.extend(img_path)
        ret_df = proc_inference(ground_truths, preds)
        ret_df["paths"] = all_paths
        return ret_df

    def _test_acc(self):
        self._set_supress_tqdm(True)
        pred_df = self.test_model(self.dls["test"])
        model_preds = pred_df[
            [e for e in pred_df if e.startswith("task_p")]
        ].values.argmax(axis=1)
        acc = balanced_accuracy_score(pred_df["task_t"], model_preds)
        return acc

    def _fit_optuna(self, trial: optuna.Trial):
        num_epochs = self.total_epochs
        best_val_loss = 90000
        val_loss = best_val_loss
        best_epoch = -1
        for i in range(num_epochs):
            self.train_epoch()
            self._log_scalar(
                "learning_rate", self.sch.get_last_lr()[0], global_step=self.c_epoch
            )
            val_loss = self.val_epoch()
            self.sch.step(val_loss)
            trial.report(val_loss, i)
            if val_loss <= best_val_loss:
                self.store_model()
                best_val_loss = val_loss
                best_epoch = i
            if (i - best_epoch) >= 5:
                print("Going to do early breaking. 5 Epochs No Progress")
                break
        return val_loss

    @classmethod
    def get_trial_suggestions(cls, trial: optuna.Trial, c_conf):
        lr = trial.suggest_categorical("learn_rate", choices=[0.01, 0.0001, 0.001])
        batch_size = trial.suggest_categorical("batch_size", choices=[64, 128, 256])
        c_conf["trainer_args"]["learn_rate"] = lr
        c_conf["trainer_args"]["batch_size"] = batch_size
        return c_conf


@TrainerRegister.register(cls_name="TwoTaskTrainer")
class TwoTaskTrainer(BasicTrainer):
    def __init__(self, model, tb_writter, conf, data_loaders):
        super().__init__(model, tb_writter, conf, data_loaders)
        self.demo_col = self.conf["col_info"]["demo_col"]

    def build_criteria(self):
        self.criterions = dict()
        get_task_incidence = ("weight_task" in self.trainer_args) and self.trainer_args[
            "weight_task"
        ]
        get_demo_incidence = ("weight_demo" in self.trainer_args) and self.trainer_args[
            "weight_demo"
        ]
        if get_task_incidence:
            ws = self.dls["train"].dataset.get_inverse_weight("task")
            task_weight = torch.tensor(ws, device=self.device, dtype=torch.float)
        else:
            task_weight = None
        if get_demo_incidence:
            ws = self.dls["train"].dataset.get_inverse_weight("demo")
            demo_weight = torch.tensor(ws, device=self.device, dtype=torch.float)
        else:
            demo_weight = None
        self.criterions["task"] = nn.CrossEntropyLoss(weight=task_weight)
        self.criterions["demo"] = nn.CrossEntropyLoss(weight=demo_weight)

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
            task_h, demo_h = self.model(img_in.to(self.device))
            task_loss = self.criterions["task"](task_h, task)
            demo_loss = self.criterions["demo"](demo_h, demo)
            loss = task_loss + demo_loss
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

    def val_epoch(self):
        self.model.eval()
        task_loss = 0
        demo_loss = 0
        with torch.no_grad():
            val_len = len(self.dls["val"])
            for i, batch in enumerate(self.dls["val"]):
                img_in = batch[self.img_col]
                task = batch[self.task_col].to(self.device)
                demo = batch[self.demo_col].to(self.device)
                task_h, demo_h = self.model(img_in.to(self.device))
                task_loss += self.criterions["task"](task_h, task).item()
                demo_loss += self.criterions["demo"](demo_h, demo).item()
            task_loss /= val_len
            demo_loss /= val_len
            self._log_scalar("val_task_loss", task_loss, global_step=self.c_epoch)
            self._log_scalar("val_demo_loss", demo_loss, global_step=self.c_epoch)
        total_loss = task_loss + demo_loss
        return total_loss

    def test_model(self, dl):
        self.model.eval()
        with torch.no_grad():
            ground_truths = defaultdict(list)
            preds = defaultdict(list)
            all_paths = list()
            for i, batch in tqdm(enumerate(dl), total=len(dl)):
                img_in = batch[self.img_col]
                task = batch[self.task_col]
                img_path = batch[f"{self.img_col}_meta_dict"]["filename_or_obj"]
                task_h, demo_h = self.model(img_in.to(self.device))
                task_h = task_h.to("cpu")
                demo_h = demo_h.to("cpu")
                truth_names = ["task_t"]
                truth_vals = [task]
                pred_names = ["task_p"]
                pred_vals = [task_h]
                for n, e in zip(truth_names, truth_vals):
                    sub = tensor_convert(e)
                    ground_truths[n].extend(sub)
                for n, e in zip(pred_names, pred_vals):
                    sub = tensor_convert(e)
                    preds[n].extend(sub)
                all_paths.extend(img_path)

        ret_df = proc_inference(ground_truths, preds)
        ret_df["paths"] = all_paths
        return ret_df

    def _test_acc(self):
        self._set_supress_tqdm(True)
        pred_df = self.test_model(self.dls["test"])
        task_model_preds = pred_df[
            [e for e in pred_df if e.startswith("task_p")]
        ].values.argmax(axis=1)
        task_acc = balanced_accuracy_score(pred_df["task_t"], task_model_preds)
        demo_model_preds = pred_df[
            [e for e in pred_df if e.startswith("demo_p")]
        ].values.argmax(axis=1)
        demo_acc = balanced_accuracy_score(pred_df["demo_t"], demo_model_preds)
        return task_acc, demo_acc


@TrainerRegister.register(cls_name="TwoTaskTrainerAux")
class TwoTaskTrainerAux(TwoTaskTrainer):
    def __init__(self, model, tb_writter, conf, data_loaders):
        super().__init__(model, tb_writter, conf, data_loaders)

    def init_optims(self):
        learn_rate = self.trainer_args["learn_rate"]
        final_params = list()
        names = list()
        for layer_name, param in self.model.named_parameters():
            if "demo" in layer_name:
                final_params.append(param)
                names.append(layer_name)
            else:
                param.requires_grad = False
        self.opti = optim.AdamW(final_params, lr=learn_rate)
        self.sch = optim.lr_scheduler.ReduceLROnPlateau(
            self.opti, mode="min", patience=3
        )


def tensor_convert(tensor):
    try:
        out = tensor.numpy()
    except:
        out = tensor
    return out
