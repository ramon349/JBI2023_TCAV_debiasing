import pandas as pd
from .helpers.args import get_tcav_args
from .models.model_factory import model_loader
from collections import OrderedDict
from .datasets.data_factory import get_dataset
from .helpers.transforms import gen_test_transforms
import torch
from types import MethodType
from captum.attr import LayerIntegratedGradients
from captum.concept import TCAV
from captum.concept import Concept
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os


def find_conv_layers(model):
    conv_layers = OrderedDict()
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            conv_layers[name] = layer
    return conv_layers


def find_relu_layers(model):
    conv_layers = OrderedDict()
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.ReLU):
            conv_layers[name] = layer
    return conv_layers


def make_concept(df, id, concept_name, loader, transforms, conf):
    """Instantiate a concept object
    - Dataframe should contain images related to one concept
    """
    ds = loader(df, transforms=transforms, conf=conf)
    data_loader = DataLoader(ds, batch_size=1, num_workers=16)
    concept = Concept(id=id, name=concept_name, data_iter=data_loader)
    return concept


def make_concepts(black_df, white_df, my_loader, transforms, conf):
    black_concept = make_concept(black_df, 0, "Black", my_loader, transforms, conf)
    white_concept = make_concept(white_df, 1, "Others", my_loader, transforms, conf)
    return black_concept, white_concept


def get_tcav_scores(cav_interp):
    scores = list()
    names = list()
    for e in cav_interp["0-1"].keys():
        names.append(e)
        scores.append(cav_interp["0-1"][e]["sign_count"][1].cpu().numpy())
    return scores, names


def main():
    conf = get_tcav_args()
    device = conf["device"][0]
    blk_df, whit_df, rem_df = make_gt_groups(conf)
    ds_cls = get_dataset(conf)
    val_transform = gen_test_transforms(conf)
    test_ds = ds_cls(rem_df, transforms=val_transform, conf=conf)
    model = model_loader(conf)
    cat = "conv"
    black_c, white_c = make_concepts(
        black_df=blk_df,
        white_df=whit_df,
        my_loader=ds_cls,
        transforms=val_transform,
        conf=conf,
    )
    if cat == "relu":
        relu_layers = find_relu_layers(model)
        layers_interest_names = list(relu_layers.keys())
    if cat == "conv":
        layers_interest = find_conv_layers(model)
        layers_interest_names = list(layers_interest.keys())
    names = layers_interest_names[
        ::10
    ]  # TODO: for viz purpose i cut down on the number of  layers
    zebra_tensors = torch.stack(
        [test_ds.__getitem__(idx).to(device) for idx in range(25)]
    )
    model.eval()
    mytcav = TCAV(
        model=model,
        layers=names,
        save_path=os.path.join(conf["log_dir"], "cav_test"),
        layer_attr_method=LayerIntegratedGradients(model, None),
    )
    tcav_scores_w_random = mytcav.interpret(
        inputs=zebra_tensors,
        experimental_sets=[[black_c, white_c]],
        processes=0,
        target=(1,),
        n_steps=2,
        additional_forward_args="demo",
    )
    tcav_scores, layer_names = get_tcav_scores(tcav_scores_w_random)
    layer_names = [".".join(e.split(".")[-2:]) for e in names]
    fig = plt.figure(dpi=300)
    plt.plot(np.hstack(tcav_scores), range(0, len(tcav_scores)))
    plt.barh(range(0, len(tcav_scores)), np.hstack(tcav_scores), align="center")
    plt.yticks(range(0, len(tcav_scores)), labels=layer_names)
    plt.ylabel("Layer Name")
    plt.xlabel("Layer TCAV Score")
    plt.title("TCAV plot Densenet fitzpatrick")
    plt.tight_layout()
    file_path = os.path.join(conf["log_dir"], "TACV_score.png")
    plt.savefig(file_path)


def make_gt_groups(conf):
    input_path = conf["csv_path"]
    val_df = pd.read_csv(input_path)
    val_df = val_df[val_df["split"] == "train"]
    val_df = val_df[val_df["fitzpatrick_scale"].isin([1, 6])]
    samples_per_concept = conf["tcav_args"]["samples_per_concept"]
    concept_test = val_df.sample(300, random_state=1996)
    rem_samples = val_df[~val_df["file"].isin(concept_test["file"])]
    black_df = (
        rem_samples[rem_samples["fitzpatrick_scale"].isin([6])]
        .copy()
        .sample(samples_per_concept)
    )
    white_df = (
        rem_samples[rem_samples["fitzpatrick_scale"].isin([1])]
        .copy()
        .sample(samples_per_concept)
    )
    # make sure samples are removed from the other dataset
    rem_samples = rem_samples[~rem_samples["file"].isin(black_df["file"].unique())]
    rem_samples = rem_samples[~rem_samples["file"].isin(white_df["file"].unique())]
    rem_samples = rem_samples[rem_samples["fitzpatrick_scale"].isin([1])]
    return black_df, white_df, rem_samples


if __name__ == "__main__":
    main()
