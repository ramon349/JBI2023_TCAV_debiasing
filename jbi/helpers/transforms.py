import torchvision.transforms as torch_trx 

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
        bright = config['brightness']
        contrast = config['contrast']
        saturation = config['saturation']
        return torch_trx.ColorJitter(brightness=(bright[0],bright[1]),contrast=contrast,saturation=saturation) 
    if names =='toTensor':
        return torch_trx.ToTensor()
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

