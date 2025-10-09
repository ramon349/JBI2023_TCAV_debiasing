from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from .data_factory import DatasetRegister
from monai.data import Dataset as monaiDataset


def make_image_d(df, cols: list):
    data_l = list()
    for i, df_row in df.iterrows():
        new_d = dict()
        for k in cols:
            new_d[k] = df_row[k]
        data_l.append(new_d)
    return data_l


@DatasetRegister.register("ImageData")
def make_image_data(transforms=None, split=None, conf=None, debug=False):
    data_path = conf["csv_path"]
    data = pd.read_csv(data_path)
    data = data[data["split"] == split]
    col_info = conf["col_info"]
    task_col = col_info["task_col"]
    img_col = col_info["img_col"]
    data_seq = make_image_d(data, cols=[task_col, img_col])
    if debug:
        data_seq = data_seq[0:200]
    return monaiDataset(data=data_seq, transform=transforms)


@DatasetRegister.register("ImageDataMask")
def make_image_data(transforms=None, split=None, conf=None, debug=False):
    data_path = conf["csv_path"]
    data = pd.read_csv(data_path)
    data = data[data["split"] == split]
    col_info = conf["col_info"]
    task_col = col_info["task_col"]
    img_col = col_info["img_col"]
    mask_col = col_info["mask_col"]
    data_seq = make_image_d(data, cols=[task_col, img_col, mask_col])
    return monaiDataset(data=data_seq, transform=transforms)


@DatasetRegister.register("TwoTask")
def skinLesionTwo(transforms=None, split=None, conf=None, debug=False):
    data_path = conf["csv_path"]
    data = pd.read_csv(data_path)
    data = data[data["split"] == split]
    col_info = conf["col_info"]
    task_col = col_info["task_col"]
    demo_col = col_info["demo_col"]
    img_col = col_info["img_col"]
    data_seq = make_image_d(data, cols=[task_col, img_col, demo_col])
    if debug:
        data_seq = data_seq[0:200]
    return monaiDataset(data=data_seq, transform=transforms)


@DatasetRegister.register("TwoTaskMask")
def skinLesionTwoMask(transforms=None, split=None, conf=None, debug=False):
    data_path = conf["csv_path"]
    data = pd.read_csv(data_path)
    data = data[data["split"] == split]
    col_info = conf["col_info"]
    task_col = col_info["task_col"]
    demo_col = col_info["demo_col"]
    img_col = col_info["img_col"]
    mask_col = col_info["mask_col"]
    data_seq = make_image_d(data, cols=[task_col, img_col, demo_col, mask_col])
    if debug:
        data_seq = data_seq[0:200]
    return monaiDataset(data=data_seq, transform=transforms)

@DatasetRegister.register(cls_name='single')
class SingleSkin(Dataset):
    def __init__(self,data,transforms=None,conf=None,debug=False):
        super().__init__()
        col_info = conf["col_info"]
        self.task_col = col_info["task_col"]
        self.demo_col = col_info["demo_col"]
        self.img_col = col_info["img_col"]
        self.mask_col = col_info["mask_col"]
        data_seq = make_image_d(data, cols=[self.task_col, self.img_col, self.demo_col, self.mask_col])
        if debug:
            data_seq = data_seq[0:200]
        self.dset = monaiDataset(data=data_seq, transform=transforms)
    def __getitem__(self, index):
        item_out = self.dset.__getitem__(index)
        return item_out[self.img_col]
    def __len__(self): 
        return len(self.dset)

