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
)
from monai.data import PILReader
from monai.transforms import MapTransform, Transform
from monai.data.meta_obj import get_track_meta
from monai.utils import convert_to_tensor
from monai.transforms.utils import (
    get_largest_connected_component_mask,
    generate_spatial_bounding_box,
)
import numpy as np


class SCrop(Transform):
    def __init__(self, update_meta=True, do_rand_shift=True,box_only=False,seed=42):
        super().__init__()
        self.update_meta = update_meta
        self.do_rand =do_rand_shift 
        self.box_only = box_only 
        self.seed= seed 
        self.generator = torch.Generator() 
        self.generator.manual_seed(seed)



    def __call__(self, vol_img, mask_img):
        
        keep_same_prob = torch.rand(size=(1,),generator=self.generator)[0]
        if self.box_only or (self.do_rand and (keep_same_prob<0.1)):
            return vol_img
        largest_island = get_largest_connected_component_mask(mask_img)
        old_box = generate_spatial_bounding_box(largest_island)
        new_box = self.make_new_bounding_box(old_box, mask_img.shape)
        x_slices = slice(new_box[0][0], new_box[1][0], 1)
        y_slices = slice(new_box[0][1], new_box[1][1], 1)
        new_img = vol_img[:, x_slices, y_slices]
        print(new_img.shape)
        return new_img
    def make_new_bounding_box(self,old_box, mask_size):
        box_center = get_box_center(old_box)
        box_size = get_box_size(old_box)
        largest = max(box_size)
        min_dim = min(mask_size[1:])
        n_square = int(np.floor(np.sqrt(min_dim)))
        if n_square > 16:
            choices = list(range(16, n_square))
        else:
            choices = [16]
        if self.do_rand: 
            choice = torch.randperm(len(choices),generator=self.generator)[0]
            selected_square =choices[0]**2 
        else: 
            selected_square = choices[0]**2

        new_center = [e + self.rand_shift_val() for e in box_center]
        shifted_box = gen_alt_box(new_center, selected_square, mask_size[1:])
        return shifted_box

    def rand_shift_val(self):
        if self.do_rand:
            val = torch.randint(low=30,high=50,size=(1,),generator=self.generator)[0]
            direction = torch.rand(size=(1,),generator=self.generator)[0]
            direction = -1 if direction< 0.5 else 1
            return val * direction
        else:
            return 0

class SCropd(MapTransform):
    def __init__(
        self, keys=None, label_key=None, allow_missing_keys=False, update_meta=False,do_rand=True
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = SCrop(update_meta=update_meta,do_rand_shift=do_rand)
        self.label_key = label_key

    def __call__(self, data):
        d = dict(data)
        old_mask_img = d[self.label_key]
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key], old_mask_img)
        return d


def get_box_size(box):
    x_dim = box[1][0] - box[0][0]
    y_dim = box[1][1] - box[0][1]
    return (x_dim, y_dim)


def get_box_center(box):
    box_size = get_box_size(box)
    x_center = box[0][0] + box_size[0] // 2
    y_center = box[0][1] + box_size[1] // 2
    return (x_center, y_center)






def gen_alt_box(ref_point, box_size, image_size):
    """
    Generate a square bounding box around a reference point that fits within the image boundaries.

    Parameters:
    - ref_point: Tuple (x, y) representing the reference point.
    - box_size: Integer representing the length of each side of the square box.
    - image_size: Tuple (width, height) representing the image dimensions.

    Returns:
    - A tuple (x_min, y_min, x_max, y_max) representing the bounding box coordinates.
    """
    x, y = ref_point
    img_width, img_height = image_size

    half_size = box_size // 2

    # Calculate initial box coordinates
    x_min = max(0, x - half_size)
    y_min = max(0, y - half_size)
    x_max = x_min + box_size
    y_max = y_min + box_size

    # Adjust if box goes out of image bounds
    if x_max > img_width:
        x_max = img_width
        x_min = max(0, x_max - box_size)
    if y_max > img_height:
        y_max = img_height
        y_min = max(0, y_max - box_size)

    return ((x_min, y_min), (x_max, y_max))
