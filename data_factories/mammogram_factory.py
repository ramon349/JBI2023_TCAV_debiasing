import numpy as np
import torch
from PIL import Image
import pandas as pd
from PIL import Image
from PIL import ImageFile
import numpy as np
import pandas as pd
import os
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True
def get_mammo_loaders(mode):
    if mode == "density":
        return mayoDensity

def pil_loader(path):
    I = Image.open(path).convert("L")
    return I

class mayoDensity(Dataset):
    def __init__(
        self,
        data_path,
        transform=None,
        loader=pil_loader,
        location="tmp",
        num_task_classes=2,
        num_demo_classes=3,
        mode="test",split='train'
    ) -> None:
        self.data = pd.read_csv(data_path, dtype="str")  # should be 2d only metadata
        self.data['tissueden']=  pd.to_numeric(self.data['tissueden'])
        self.data = self.data[self.data['split']==split].copy()
        self.data['crop_idx']= self.data['crop_idx'].apply(lambda x: eval(x))
        self.loader = loader
        self.transform = transform
        self.num_task_classes = num_task_classes
        self.num_demo_classes = num_demo_classes
        self.race_dict = {'African American  or Black':0, 'Caucasian or White':1, 'Asian':2, 'Other':3}
        self.mode = mode

    def load_img(self, study):
        img = pil_loader(study['png_path'])
        img = np.array(img)
        print(f" IMAGE TYPE IS : {img.dtype}")
        crop = study['crop_idx']
        img =  img[crop[1]:crop[1]+crop[3],crop[0]:crop[0]+crop[2] ].copy()
        img =  Image.fromarray(img)
        raw_img = pil_to_tensor(img).to(torch.float)
        if self.transform:
            img = self.transform(raw_img)
        return img

    def __getitem__(self, idx):
        study = self.data.iloc[idx]
        label = study['tissueden']
        demo_info = self.race_dict[study["ETHNICITY_DESC"]]
        img = self.load_img(study)
        return (img, label, demo_info, study['png_path'])

    def __len__(self):
        return self.data.shape[0]