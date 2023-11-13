import numpy as np
import torch

# from skimage.filters.rank import entropy
# from skimage.morphology import disk
# from skimage import filters

class toGPU(object):
    def __init__(self,gpu=0): 
        pass 
    def __call__(self,img): 
        return img.to('cuda:0').copy()
    
class make8bit(object):
    def __init__(self,mode='sample'): 
        pass 
    def __call__(self,img): 
        x=   np.array(img)
        x_scaled =  (np.maximum(x,0) /x.max() ) * 255.0
        x_scaled = x_scaled.astype(np.uint8)
        return torch.tensor(x_scaled)


class myRep(object):
    def __init__(self, name="hi"):
        self.name = name

    def __call__(self, tensor):
        return tensor.repeat(3, 1, 1)


# stolen from forums.  wil add some modds later.
# idea.  same gaussian across 3 laeyrs or different noise kjk
class AddGaussNoise(object):
    def __init__(self, p=0.5, mean=0, std=0.005):
        self.p = p
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        if torch.randn(1) < self.p:
            out = (
                tensor + torch.randn(tensor.size()) * self.std + self.mean
            )  # you changed this to be compatible with the 3.7 version. 3.8 somehow requires .size()
            return out.type(torch.FloatTensor)
        else:
            return tensor.type(torch.FloatTensor)

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class AddEntropySobel(object):
    def __init__(self, mode="sobel"):
        self.mode = mode
        self.disk = disk(10)

    def __call__(self, tensor):
        tensor = np.array(tensor)
        feat = entropy(tensor, self.disk)
        if self.mode == "sobel":
            feat2 = filters.sobel(tensor)
        else:
            feat2 = filters.scharr(tensor)
        return np.stack((tensor, feat, feat2), axis=-1)  # this is highly sus

    def __repr__(self):
        return self.__class__.__name__ + "(Entropy(disk ={}),sobel )".format(self.disk)


class typeConv(object):
    def __init__(self, typ=torch.FloatTensor):
        self.typ = typ

    def __call__(self, tensor):
        return tensor.type(self.typ)

    def __repr__(self):
        return "conv to float"
