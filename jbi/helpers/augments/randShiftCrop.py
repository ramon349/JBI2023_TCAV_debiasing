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
    def __init__(self, update_meta=True, do_rand=True):
        super().__init__()
        self.update_meta = update_meta
        self.do_rand = do_rand

    def __call__(self, vol_img, mask_img):
        if np.random.random() < 0.01:
            return vol_img
        largest_island = get_largest_connected_component_mask(mask_img)
        old_box = generate_spatial_bounding_box(largest_island)
        new_box = make_new_bounding_box(old_box, mask_img.shape, do_rand=self.do_rand)
        x_slices = slice(new_box[0][0], new_box[1][0], 1)
        y_slices = slice(new_box[0][1], new_box[1][1], 1)
        new_img = vol_img[:, x_slices, y_slices]
        return new_img


class SCropd(MapTransform):
    def __init__(
        self, keys=None, label_key=None, allow_missing_keys=False, update_meta=False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = SCrop(update_meta=update_meta)
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


def rand_shift_val(do_rand):
    if do_rand:
        val = np.random.randint(low=30, high=50)
        direction = -1 if np.random.rand() < 0.5 else 1
        return val * direction
    else:
        return 0


def make_new_bounding_box(old_box, mask_size, do_rand=True):
    box_center = get_box_center(old_box)
    box_size = get_box_size(old_box)
    largest = max(box_size)
    min_dim = min(mask_size[1:])
    n_square = int(np.floor(np.sqrt(min_dim)))
    if n_square > 16:
        choices = list(range(16, n_square))
    else:
        choices = [16]
    selected_square = np.random.choice(choices) ** 2

    new_center = [e + rand_shift_val(do_rand) for e in box_center]
    shifted_box = gen_alt_box(new_center, selected_square, mask_size[1:])
    return shifted_box


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
