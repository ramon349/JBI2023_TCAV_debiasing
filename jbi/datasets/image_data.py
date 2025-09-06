from torch.utils.data import Dataset 
from PIL import Image 
import pandas as pd  
from .data_factory import DatasetRegister 

@DatasetRegister.register("ImageData")
class skinLesion(Dataset):
    def __init__(self,transforms=None,split=None,conf=None) -> None:
        data_path = conf['csv_path']
        self.data = pd.read_csv(data_path,dtype='str')
        self.data = self.data[self.data['split']==split]  
        self.transform = transforms
        col_info = conf['col_info']
        self.img_col = col_info['img_col']  
        self.task_col = col_info['task_col']  
        self.data[self.task_col] = pd.to_numeric(self.data[self.task_col])
    def load_img(self,study): 
        my_img  = Image.open(study[self.img_col])
        if my_img.mode != "RGB": 
            my_img = my_img.convert("RGB")
        return my_img 
    def __getitem__(self, index) :
        study = self.data.iloc[index]
        img = self.load_img(study)
        label =  study[self.task_col] 
        if self.transform:
            img = self.transform(img)
        return img ,label,study[self.img_col]
    def __len__(self):
        return self.data.shape[0]
    def get_inverse_weight(self,mode): 
        weights = list() 
        if mode=='task': 
            total_elems = self.data.shape[0]
            for uni_label in self.data[self.task_col].unique(): 
                weights.append((self.data[self.task_col]==uni_label).sum()/total_elems)
            return weights
        else: 
            return None 
            


                 
@DatasetRegister.register("TwoTask")
class skinLesionTwo(skinLesion):
    def __init__(self, transforms=None, split=None, conf=None):
        super().__init__(transforms, split, conf)
        col_info = conf['col_info'] 
        self.demo_col = col_info['demo_col']
        self.data[self.demo_col] = pd.to_numeric(self.data[self.demo_col])
    def __getitem__(self, index): 
        img,label,path = super().__getitem__(index)
        study = self.data.iloc[index]
        demo_label =  study[self.demo_col]  
        return img,label,demo_label,path 
    def get_inverse_weight(self,mode): 
        weights = list() 
        if mode=='task': 
            total_elems = self.data.shape[0]
            for uni_label in self.data[self.task_col].unique(): 
                weights.append((self.data[self.task_col]==uni_label).sum()/total_elems)
            return weights
        if mode=='demo':
            total_elems = self.data.shape[0]
            for uni_label in self.data[self.demo_col].unique(): 
                weights.append((self.data[self.demo_col]==uni_label).sum()/total_elems)
            return weights
        else: 
            return None 
        


@DatasetRegister.register("single")
class skinSingle(skinLesion): 
    def __init__(self,data, transforms=None, split=None, conf=None):
        self.data = data
        self.transform = transforms
        col_info = conf['col_info']
        self.img_col = col_info['img_col']  
        self.task_col = col_info['task_col']  
        self.data[self.task_col] = pd.to_numeric(self.data[self.task_col])
    def load_img(self,study): 
        my_img  = Image.open(study[self.img_col])
        if my_img.mode != "RGB": 
            my_img = my_img.convert("RGB")
        return my_img 
    def __getitem__(self, index) :
        study = self.data.iloc[index]
        img = self.load_img(study)
        label =  study[self.task_col] 
        if self.transform:
            img = self.transform(img)
        return img
    def __len__(self):
        return self.data.shape[0]
    
