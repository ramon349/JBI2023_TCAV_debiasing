import torch
from torch.nn import functional as F
from monai.transforms import (
    RandScaleCropd,
    ResizeD,
    RandGaussianNoiseD,
    NormalizeIntensityD,
    ScaleIntensityD,
    RandRotate90D,
    LoadImageD,
    Compose,
    EnsureChannelFirstd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RepeatChanneld
)
from monai.data import PILReader, PydicomReader 
from monai.transforms import MapTransform, Transform
from monai.data.meta_obj import get_track_meta
from monai.utils import convert_to_tensor
from .augments.randShiftCrop import SCropd
import pydicom as pyd 
from pydicom.pixels import apply_voi_lut
import torch
from monai.utils import convert_to_dst_type
def get_img_reader(conf): 
    reader_name = conf['img_reader'] 
    match reader_name:
        case 'png': 
            return PILReader()
        case 'dcm': 
            return  PydicomReader(prune_metadata=True,affine_lps_to_ras=False,swap_ij=False)
        case _: 
            raise ValueError("No propoer Image reader name was provided")

def get_transform(names, main_config):
    """Given the name of a transform we build said torch transform. Using params from config as needed
    names: str specifying which transfrom to build
    config: dict containing parameters used by all transforms
    """
    config = main_config["transform_conf"]
    useMask = config["use_mask"]
    col_info = main_config["col_info"]
    img_col = col_info["img_col"]
    if useMask:
        mask_col = col_info["mask_col"]
        keys = [img_col, mask_col]
    else:
        keys = [img_col]
    match names:
        case "load":
            reader_obj = get_img_reader(config)
            return LoadImageD(keys=keys, reader=reader_obj, image_only=False)
        case "scaleIntensity":
            return ScaleIntensityD(keys=[img_col])
        case "norm":
            mu = torch.tensor(config["norm_mu"])
            std = torch.tensor(config["norm_std"])
            return NormalizeIntensityD(
                keys=[img_col], subtrahend=mu, divisor=std, channel_wise=True
            )
        case "channelFirst":
            return EnsureChannelFirstd(keys=keys)
        case "randScaleCrop":
            scale_min, scale_max = config["ScaleCropMinMax"]
            return RandScaleCropd(
                keys=keys,
                roi_scale=scale_min,
                max_roi_scale=scale_max,
                random_size=True,
                random_center=False,
            )
        case "randCropPos":
            return RandCropByPosNegLabeld(
                keys=keys,
                label_key=mask_col,
                pos=255,
                neg=0,
                num_samples=1,
                image_key=img_col,
                spatial_size=(-1, -1),
            )
        case "resize":
            shape0 = config["img_shape"][0]
            shape1 = config["img_shape"][1]
            return ResizeD(keys=keys, spatial_size=[shape0, shape1])
        case "rotate":
            return RandRotate90D(keys=keys, prob=0.5,max_k=3)
        case "randGaus":
            return RandGaussianNoiseD(keys=[img_col], mean=0, std=0.1, prob=0.5)
        case "make8Bit":
            return Make8Bitd(keys=[img_col])
        case "flip": 
            return  RandFlipd(keys=keys,spatial_axis=[0,1])
        case "Make3Channel":
            return  RepeatChanneld(keys=[img_col],repeats=3)
        case "SCrop":
            if 'do_rand_shift' in config:
                do_rand_shift = config['do_rand_shift']
            else: 
                do_rand_shift = True
            return SCropd(keys=[img_col, mask_col], label_key=mask_col,do_rand=do_rand_shift)
        case "ApplyVoi": 
            return ApplyVoiD(keys=[img_col])
    raise Exception(f"Couldn't fnd a match for argument {names}")


def gen_transforms(confi):
    train_transform = Compose(
        [get_transform(e, confi) for e in confi["train_transforms"]]
    )
    val_transform = Compose([get_transform(e, confi) for e in confi["test_transforms"]])
    return train_transform, val_transform


def gen_test_transforms(confi, mode="test"):
    my_transforms = list()
    for e in confi["test_transforms"]:
        if e == "labelMask" and mode == "infer":
            continue
        l_transform = get_transform(e, confi)
        print(l_transform)
        my_transforms.append(l_transform)
    val_transform = Compose(
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

    def __init__(self, size, fill=0, padding_mode="constant"):
        """
        Initializes the transform.

        Args:
            size (tuple): The desired output size (height, width).
            fill (int or tuple): Pixel fill value for constant padding. Default is 0.
            padding_mode (str): Type of padding. Should be 'constant', 'edge',
                                'reflect' or 'symmetric'. Default is 'constant'.
        """
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError(
                "Size must be a tuple or list of two integers (height, width)"
            )

        self.target_height, self.target_width = size
        self.fill = fill
        self.padding_mode = padding_mode

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

        max_c = max(current_height, current_width)
        target_c = self.target_height
        if max_c > self.target_height:
            target_c = max_c

        # Calculate padding
        pad_height = target_c - current_height
        pad_width = target_c - current_width
        # Only pad if the image is smaller than the target size
        if pad_height < 0:
            pad_height = 0
        if pad_width < 0:
            pad_width = 0

        # Calculate padding for each side (left, top, right, bottom)
        # This ensures the image is centered
        fill_ratios = torch.rand(size=(1,))
        pad_top = int(pad_height * fill_ratios)
        # pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        total_pad = pad_bottom + pad_top
        pad_diff = pad_height - total_pad
        if pad_diff:
            if torch.rand(size=(1,)) < 0.5:
                pad_bottom += pad_diff
            else:
                pad_height += pad_diff
        pad_left = int(pad_width * fill_ratios)
        pad_right = pad_width - pad_left
        total_pad = pad_left + pad_right
        pad_diff = pad_width - total_pad
        if pad_diff:
            if torch.rand(size=(1,)) < -0.5:
                pad_left += pad_diff
            else:
                pad_right += pad_diff
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        # Apply padding
        out_pad = F.pad(img, padding, value=self.fill, mode=self.padding_mode)
        return out_pad

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(size=({self.target_height}, {self.target_width}), padding_mode={self.padding_mode}, fill={self.fill})"
        )


class AddGaussNoise(object):
    def __init__(self, p=0.5, mean=0, std=0.15):
        self.p = p
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        if torch.randn(1) < self.p:
            noise_vec = torch.randn(tensor.size()) * self.std + self.mean
            noise_vec = noise_vec * (tensor.max() / 2)

            out = (
                tensor + noise_vec
            )  # you changed this to be compatible with the 3.7 version. 3.8 somehow requires .size()
            return out.type(torch.FloatTensor)
        else:
            return tensor.type(torch.FloatTensor)

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class Make8Bit(Transform):
    def __init__(self, update_meta=True):
        super().__init__()
        self.update_meta = update_meta

    def __call__(self, img):
        img = convert_to_tensor(img, track_meta=get_track_meta()).squeeze(0)
        x_min = torch.min(img)
        if x_min < 0:
            img = img + (-1 * x_min)
        else:
            img = img - x_min
        img_scaled = (img / img.max()) * 255
        return img_scaled


class Make8Bitd(MapTransform):
    def __init__(self, keys=None, allow_missing_keys=False, update_meta=False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = Make8Bit(update_meta=update_meta)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class ApplyVoi(Transform):
    def __init__(self, update_meta=True):
        super().__init__()
        self.update_meta = update_meta

    def __call__(self,img,meta_dict):
        og_file = meta_dict['filename_or_obj'] 
        old_dcm = pyd.dcmread(og_file,stop_before_pixels=True)
        out_img = apply_voi_lut(img.numpy(),ds=old_dcm)
        new_meta = convert_to_dst_type(out_img,img) [0].unsqueeze(0)
        return new_meta 
class ApplyVoiD(MapTransform): 
    def __init__(self, keys, allow_missing_keys = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ApplyVoi(update_meta=False)
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            meta_d_name =  f"{key}_meta_dict"
            try: 
                d[key] = self.converter(d[key],d[meta_d_name])
            except  KeyError: 
                d[key]= d[key]
            if '00280030' in d[meta_d_name]: 
                del d[meta_d_name]['00280030']
        return d
