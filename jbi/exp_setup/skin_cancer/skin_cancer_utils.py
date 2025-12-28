import pandas as pd
from sklearn.model_selection import train_test_split
import os
from argparse import ArgumentParser
from PIL import UnidentifiedImageError, Image
from multiprocessing import Pool
from tqdm import tqdm


def check_img(s):
    is_file = True
    try:
        Image.open(s)
    except UnidentifiedImageError:
        is_file = False
    except FileNotFoundError:
        is_file = False
    return s, is_file


def _make_args():
    args = ArgumentParser()
    args.add_argument("--mode", type=str, required=True, choices=["download_fitz"])
    args.add_argument("--output_csv", type=str, required=True)
    args.add_argument("--local_csv_path", type=str, required=False, default=None)
    args.add_argument("--data_root", type=str, required=True)
    args.add_argument("--mask_data_root", type=str, required=True)
    return vars(args.parse_args())


def bin_fitz(x): 
    x = int(x)
    if  x <=2: 
        return 0 
    if x <=4: 
        return 1 
    else: 
        return 2
def pull_dataset(
    save_path: str, local_csv_path=None, data_root=None, mask_data_root=None
):
    """Downloads the fitzparick17 dataset and saves it as necessary"""
    if local_csv_path is None:
        print(f"Downloading from github")
        df = pd.read_csv(
            "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/refs/heads/main/fitzpatrick17k.csv"
        )
    else:
        df = pd.read_csv(local_csv_path)

    df["file"] = df["md5hash"].apply(lambda x: os.path.join(data_root, f"{x}.jpg"))
    df["mask_file"] = df["md5hash"].apply(
        lambda x: os.path.join(mask_data_root, f"{x}.jpg")
    )
    stat_map = dict()
    with Pool(10) as P:
        res = P.imap_unordered(check_img, df["file"])
        for k, v in tqdm(res, total=df.shape[0]):
            stat_map[k] = v
    df["is_file"] = df["file"].map(stat_map)
    df = df[df["is_file"]].copy()
    df = df[df["fitzpatrick_scale"] >= 0]
    df["discrete_fitz"] = (df["fitzpatrick_scale"].map({1:0,2:0,3:1,4:1,5:2,6:2})).astype(int)
    df['fitz_cat'] =  df['fitzpatrick_scale'].apply(bin_fitz)
    # [df = df[df['three_partition_label'].isin(["malignant","benign"])].copy()

    print(f"Stratiyin by partition label")
    tr, val = train_test_split(
        df,
        random_state=42,
        shuffle=True,
        train_size=0.60,
    )
    val, ts = train_test_split(
        val, random_state=42, stratify=val["three_partition_label"], train_size=0.5
    )
    tr["split"] = "train"
    val["split"] = "val"
    ts["split"] = "test"
    final_df = pd.concat([tr, val, ts])
    final_df["three_partition_label_cls"] = final_df["three_partition_label"].map(
        {"non-neoplastic": 0, "malignant": 1, "benign": 0}
    )
    print(f"Saving dataset file to {save_path}")
    final_df.to_csv(save_path, index=False)
    return final_df


def _skin_transform_params() -> dict[str, list[str] | dict[str, list[float]]]:
    conf_params = {
        "train_transforms": [
            "load",
            "channelFirst",
            "SCrop",
            "scaleIntensity",
            "norm",
            "resize",
            "flip",
            "rotate",
            "randGaus",
        ],
        "test_transforms": ["load", "channelFirst", "scaleIntensity", "norm", "resize"],
        "transform_conf": {
            "use_mask": 1,
            "norm_mu": [0.485, 0.456, 0.406],
            "norm_std": [0.229, 0.224, 0.225],
            "brightness": [0.8, 1.2],
            "saturation": [0.8, 1.1],
            "contrast": [0.8, 1.2],
            "img_shape": [224, 224],
            "ScaleCropMinMax": [0.5, 1.2],
        },
    }
    return conf_params


def main():
    conf = _make_args()
    mode = conf['mode']
    match mode:
        case "download_fitz":
            pull_dataset(
                conf["output_csv"],
                local_csv_path=conf["local_csv_path"],
                data_root=conf["data_root"],
                mask_data_root=conf["mask_data_root"],
            )
        case _:
            raise ValueError("Illegal Mode argument")


if __name__ == "__main__":
    main()
