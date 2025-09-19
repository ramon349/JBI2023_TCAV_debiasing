from torch.utils.data import Dataset 
from PIL import Image 
import pandas as pd  
from .data_factory import DatasetRegister 
from monai.data import Dataset  as monaiDataset


def make_image_d(df,cols:list): 
    data_l = list() 
    for i,df_row in df.iterrows(): 
        new_d = dict()
        for k in cols: 
            new_d[k] = df_row[k]
        data_l.append(new_d) 
    return data_l 

@DatasetRegister.register("ImageData")
def make_image_data(transforms=None,split=None,conf=None,debug=False): 
        data_path = conf['csv_path']
        data = pd.read_csv(data_path)
        data = data[data['split']==split]  
        col_info = conf['col_info']
        task_col = col_info['task_col']
        img_col =  col_info['img_col']
        data_seq = make_image_d(data,cols=[task_col,img_col])
        if debug: 
            data_seq = data_seq[0:200]
        return monaiDataset(data=data_seq,transform=transforms)

@DatasetRegister.register("ImageDataMask")
def make_image_data(transforms=None,split=None,conf=None,debug=False): 
        data_path = conf['csv_path']
        data = pd.read_csv(data_path)
        data = data[data['split']==split]  
        col_info = conf['col_info']
        task_col = col_info['task_col']
        img_col =  col_info['img_col']
        mask_col = col_info['mask_col']
        data_seq = make_image_d(data,cols=[task_col,img_col,mask_col])
        return monaiDataset(data=data_seq,transform=transforms)

                 
@DatasetRegister.register("TwoTask")
class skinLesionTwo():
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
class skinSingle(skinLesionTwo): 
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
    
