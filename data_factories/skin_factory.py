import random
from tkinter import W 
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader 
import pandas as pd
from torchvision import transforms,io 
import numpy as np
from torch_transforms import *
from PIL import Image 
from torchvision.transforms.functional import pil_to_tensor
import pdb 
def get_skin_loaders(mode):
    if mode == "skinBenign":
        return skinLesion
    if mode == 'single': 
        return skinsingle
    if mode=='skinBias':
        return skinBias
    if mode=='skinColor':
        return skinFiz
    if mode =='skinBenignBias':
        return skinBenignBias
class skinLesion(Dataset):
    def __init__(self,data_path,transform=None,split=None,num_task_classes=3) -> None:
        self.data = pd.read_csv(data_path,dtype='str')
        self.data = self.data[self.data['split']==split] 
        self.data = self.data.sample(frac=0.1)
        self.transform = transform
        self.label_map = {'benign':0,'malignant':1 ,'non-neoplastic':2}
        self.num_task_classes = num_task_classes
        self.class_names = ['benign','malignant','non-neoplastic']
        self.data = self.data[self.data['three_partition_label']!='non-neoplastic']
    def load_img(self,study): 
        my_img  = Image.open(study['file'])
        if my_img.mode != "RGB": 
            my_img = my_img.convert("RGB")
        return my_img 
    def __getitem__(self, index) :
        study = self.data.iloc[index]
        img = self.load_img(study)
        label = self.label_map[study['three_partition_label']]
        if self.transform:
            img = self.transform(img)
        return img ,label  , study['file']
    def __len__(self):
        return self.data.shape[0]
class skinsingle(skinLesion):
    def __init__(self, data, transform=None, split=None, num_task_classes=3) -> None:
        self.data = data
        self.data = self.data.sample(frac=0.1)
        self.transform = transform
        self.label_map = {'benign':0,'malignant':1 ,'non-neoplastic':2}
        self.num_task_classes = 3 
        self.class_names = ['benign','malignant','non-neoplastic']
    def __getitem__(self, index):
        return super().__getitem__(index)[0]

class skinBias(Dataset):
    def __init__(self,data_path,transform=None,split=None,num_task_classes=3) -> None:
        self.data = pd.read_csv(data_path,dtype='str')
        self.data = self.data[self.data['split']==split]  
        self.data = self.data.sample(frac=0.1)
        self.data['fitz'] = self.data['fitzpatrick'].apply(lambda x: 0 if int(x)<=3 else 1)
        self.transform = transform
        self.label_map = {'benign':0,'malignant':1 ,'non-neoplastic':2}
        self.num_task_classes = 2 
        self.num_dem_classes = 2
        self.class_names = ['benign','malignant','non-neoplastic']
        self.dem_class_names = ["light","dark"]
        self.data = self.data[self.data['three_partition_label']!='non-neoplastic']
    def load_img(self,study): 
        my_img  = Image.open(study['file'])
        if my_img.mode != "RGB": 
            my_img = my_img.convert("RGB")
        return my_img 
    def __getitem__(self, index) :
        study = self.data.iloc[index]
        img = self.load_img(study) 
        label = self.label_map[study['three_partition_label']]
        demo_label = study['fitz']
        if self.transform:
            img = self.transform(img)
        return img ,label,demo_label, study['file']
    def __len__(self):
        return self.data.shape[0]

class skinBenignBias(skinBias):
    def __init__(self, data_path, transform=None, split=None, num_task_classes=2) -> None:
        super().__init__(data_path, transform, split, num_task_classes=num_task_classes)
        self.num_task_classes = 2
        self.num_demo_classes = 2

class skinFiz(Dataset):
    def __init__(self,data_path,transform=None,split=None,num_task_classes=3) -> None:
        self.data = pd.read_csv(data_path,dtype='str')
        self.data = self.data[self.data['split']==split] 
        self.data['fitzpatrick'] = pd.to_numeric(self.data['fitzpatrick']) 
        self.data = self.data[self.data['fitzpatrick']>0] 
        self.data = self.data.sample(frac=0.1)
        self.transform = transform
        self.num_task_classes = 7
        self.num_dem_classes = 2
        self.class_names = ['benign','malignant','non-neoplastic']
    def load_img(self,study): 
        my_img  = Image.open(study['file']) 
        if my_img.mode != "RGB": 
            my_img = my_img.convert('RGB')
        x =   pil_to_tensor(my_img)
        return x
    def __getitem__(self, index) :
        study = self.data.iloc[index]
        img = self.load_img(study) 
        demo_label = study['fitzpatrick']
        lbl = study['three_partition_label']
        if self.transform:
            img = self.transform(img)
        return img ,lbl,demo_label, study['file']
    def __len__(self):
        return self.data.shape[0]
