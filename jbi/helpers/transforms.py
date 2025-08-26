import torchvision.transforms as torch_trx
from torchvision.transforms.v2 import ScaleJitter,ToImage,ToTensor,ToDtype
import torch
from torch.nn import functional as F 

def get_transform(names, main_config):
    """Given the name of a transform we build said torch transform. Using params from config as needed
    names: str specifying which transfrom to build
    config: dict containing parameters used by all transforms
    """
    config = main_config["transform_conf"]
    if names == "norm":
        mu = config["norm_mu"] = config["norm_mu"]
        std = config["norm_std"] = config["norm_std"]
        return torch_trx.Normalize(mu, std)
    if names == "resize":
        shape0 = config["img_shape"][0]
        shape1 = config["img_shape"][1]
        return torch_trx.Resize((shape0, shape1))
    if names == "horizontal":
        return torch_trx.RandomHorizontalFlip(p=0.5)
    if names == "vertical":
        return torch_trx.RandomVerticalFlip(p=0.5)
    if names == "affine":
        return torch_trx.RandomAffine(15)
    if names == "centerCrop":
        return torch_trx.CenterCrop((224, 224))
    if names == "ColorJitter":
        bright = config["brightness"]
        contrast = config["contrast"]
        saturation = config["saturation"]
        return torch_trx.ColorJitter(
            brightness=(bright[0], bright[1]), contrast=contrast, saturation=saturation
        )
    if names == "toTensor":
        return ToDtype(dtype=torch.float32,scale=True)
    if names =='ScaleJitter':
        print('adding')
        return  ScaleJitter(target_size=(224,224),scale_range=(0.8,1.5),)
    if names=='toImage':
        return ToImage()
    if names=='pad':
        return torch_trx.Pad()
    if names=='padSquare': 
        return PadToSize(size=(224,224))
    raise Exception(f"Couldn't fnd a match for argument {names}")


def gen_transforms(confi):
    train_transform = torch_trx.Compose(
        [get_transform(e, confi) for e in confi["train_transforms"]]
    )
    val_transform = torch_trx.Compose(
        [get_transform(e, confi) for e in confi["test_transforms"]]
    )
    return train_transform, val_transform


def gen_test_transforms(confi, mode="test"):
    my_transforms = list()
    for e in confi["test_transforms"]:
        if e == "labelMask" and mode == "infer":
            continue
        l_transform = get_transform(e, confi)
        print(l_transform)
        my_transforms.append(l_transform)
    val_transform = torch_trx.Compose(
        my_transforms
    )  # Compose([get_transform(e, confi) for e in confi["test_transforms"]])
    return val_transform

class PadToSize:
    """
    Pads a tensor or PIL image to a specified height and width.

    This transform calculates the required padding to center the input image
    within the target dimensions. If the input image is larger than the
    target size in any dimension, it is not padded in that dimension.
    """

    def __init__(self, size, fill=0, padding_mode='constant'):
        """
        Initializes the transform.

        Args:
            size (tuple): The desired output size (height, width).
            fill (int or tuple): Pixel fill value for constant padding. Default is 0.
            padding_mode (str): Type of padding. Should be 'constant', 'edge',
                                'reflect' or 'symmetric'. Default is 'constant'.
        """
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError("Size must be a tuple or list of two integers (height, width)")
        
        self.target_height, self.target_width = size
        self.fill = fill
        self.padding_mode = padding_mode
        print(self.padding_mode)

    def __call__(self, img):
        """
        Applies the padding to the image.

        Args:
            img (PIL Image or Tensor): Image to be padded.

        Returns:
            PIL Image or Tensor: Padded image.
        """
        # Get current image size
        if isinstance(img, torch.Tensor):
            # Tensor shape is (C, H, W)
            current_height, current_width = img.shape[-2:]
        elif isinstance(img, Image.Image):
            # PIL Image size is (width, height)
            current_width, current_height = img.size
        else:
            raise TypeError("Input must be a PIL Image or a torch.Tensor")

        max_c = max(current_height,current_width)
        target_c = self.target_height
        if max_c > self.target_height: 
            target_c = max_c 

        # Calculate padding
        pad_height = target_c - current_height
        pad_width = target_c - current_width
        print(f"{pad_height},{pad_width}")
        # Only pad if the image is smaller than the target size
        if pad_height < 0:
            pad_height = 0
        if pad_width < 0:
            pad_width = 0

        # Calculate padding for each side (left, top, right, bottom)
        # This ensures the image is centered
        fill_ratios = torch.rand(size=(1,))
        pad_top = int(pad_height*fill_ratios)
        #pad_top = pad_height // 2
        pad_bottom = (pad_height - pad_top) 
        total_pad = pad_bottom + pad_top
        pad_diff =  pad_height - total_pad
        if  pad_diff: 
            if torch.rand(size=(1,)) <0.5: 
                pad_bottom += pad_diff 
            else: 
                pad_height +=pad_diff
        pad_left = int(pad_width*fill_ratios)
        pad_right = pad_width - pad_left 
        total_pad = pad_left + pad_right 
        pad_diff = pad_width - total_pad
        if pad_diff: 
            if torch.rand(size=(1,))<-0.5: 
                pad_left += pad_diff 
            else: 
                pad_right += pad_diff 
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        # Apply padding
        out_pad = F.pad(img, padding,value=self.fill,mode=self.padding_mode)
        return out_pad

    def __repr__(self):
        return self.__class__.__name__ + f'(size=({self.target_height}, {self.target_width}), padding_mode={self.padding_mode}, fill={self.fill})'

