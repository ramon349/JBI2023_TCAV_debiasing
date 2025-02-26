import torchvision.transforms as torch_trx 
import torchvision.transforms as torch_trx
from skimage.measure import label, regionprops_table
import pandas as pd
import torch


class CropMammo(object):
    def __init__(self) -> None:
        super(CropMammo, self).__init__()

    def __call__(self, tens):
        arr = tens.numpy()
        zero_mask = arr > arr.min()
        lbl = label(zero_mask)
        props = pd.DataFrame(
            regionprops_table(lbl, properties=["label", "bbox", "area"])
        )
        props = props.sort_values(by="area", ascending=False)
        row = props.iloc[0]
        rmin, rmax, cmin, cmax = row[[f"bbox-{e}" for e in [1, 4, 2, 5]]].astype(
            int
        )  # BEWARE OF THE BATCH DIMENSION
        new_tens = tens[:, rmin:rmax, cmin:cmax]
        return new_tens


def get_transform(names,main_config): 
    """ Given the name of a transform we build said torch transform. Using params from config as needed 
    names: str specifying which transfrom to build 
    config: dict containing parameters used by all transforms  
    """
    config = main_config['transform_conf'] 
    if names=='norm': 
        mu = config['norm_mu']=config['norm_mu']
        std = config['norm_std']=config['norm_std']
        return torch_trx.Normalize(mu,std)
    if names =='resize':
        shape0 = config['img_shape'][0]
        shape1 = config['img_shape'][1]
        return torch_trx.Resize((shape0,shape1))
    if names =='horizontal':
        return torch_trx.RandomHorizontalFlip(p=0.5)
    if names =='vertical':
        return torch_trx.RandomVerticalFlip(p=0.5)
    if names=='affine':
        return torch_trx.RandomAffine(15)
    if names =='centerCrop':
        return torch_trx.CenterCrop((224,224))
    if names == 'ColorJitter':
        if 'brightness' in config: 
            bright = config['brightness']
        if 'contrast' in config:
            contrast = config['contrast']
        if 'saturation' in config: 
            saturation = config['saturation']
        return torch_trx.ColorJitter(brightness=(bright[0],bright[1]),contrast=contrast,saturation=saturation) 
    if names =='toTensor':
        return torch_trx.ToTensor()
    if names =='crop':
        return CropMammo()
    raise Exception(f"Couldn't fnd a match for argument {names}")

def gen_transforms(confi):
    train_transform = torch_trx.Compose(
        [get_transform(e, confi) for e in confi["train_transforms"]]
    )
    val_transform = torch_trx.Compose([get_transform(e, confi) for e in confi["test_transforms"]])
    return train_transform, val_transform


def gen_test_transforms(confi,mode='test'):
    my_transforms = list() 
    for e in confi['test_transforms']:
        if e =='labelMask' and mode=='infer':
            continue 
        l_transform = get_transform(e,confi)
        print(l_transform)
        my_transforms.append(l_transform)
    val_transform = torch_trx.Compose(my_transforms) #Compose([get_transform(e, confi) for e in confi["test_transforms"]])
    return val_transform

