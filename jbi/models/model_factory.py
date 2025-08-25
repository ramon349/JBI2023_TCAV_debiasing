import torch
import pdb
from collections import OrderedDict
from torch import nn
from torchvision.models import densenet121, DenseNet121_Weights


class ModelRegister:
    __data = {}

    @staticmethod
    def __models():
        if not hasattr(ModelRegister, "_data"):
            ModelRegister._data = {}
        return ModelRegister._data

    @classmethod
    def register(cls, cls_name=None):
        def decorator(cls_obj):
            cls.__data[cls_name] = cls_obj
            return cls_obj

        return decorator

    @classmethod
    def get_model(cls, key):
        return cls.__data[key]

    @classmethod
    def num_models(cls):
        return len(cls.__data)

    @classmethod
    def get_models(cls):
        return cls.__data.keys()


def remove_module(w_d):
    new_d = OrderedDict()
    for k, v in w_d.items():
        new_name = k.replace("module.", "")
        new_d[new_name] = v
    return new_d


def model_loader(conf):
    model_params = conf["model_parameters"]
    model = ModelRegister.get_model(conf["model"])(model_params)
    if "model_weight" in conf and conf["model_weight"]:
        model_w = torch.load(conf["model_weight"], map_location="cpu")
        model_w = model_w["model_weights"]
        model_w = remove_module(model_w)
        model.load_state_dict(model_w)
        print(f"Loaded state dict")
    model = model.to(conf["device"][0])
    return model


@ModelRegister.register("densenet121")
class myDensenet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        num_classes = conf["num_task"]
        o_feats = self.model.classifier.in_features
        self.model.classifier = nn.Linear(o_feats, num_classes)

    @staticmethod
    def get_trial_suggestions(trial_obj, model_params):
        return model_params

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    print(f"We have {ModelRegister.num_models()} models")
    print(f"They are {ModelRegister.get_models()}")
