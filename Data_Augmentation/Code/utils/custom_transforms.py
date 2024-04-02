from __future__ import absolute_import
from typing import Any,List,Literal

from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch
import matplotlib.patches as patches

    
class HideAndSeek(object):
    """
    Summary:
        Hide-and-seek augmentaion
    
    """
    def __init__(self, probability = 0.5,grid_ratio=0.25,patch_probabilty=0.5, mean=[0.4914, 0.4822, 0.4465],value:Literal['M','R','Z'] = "Z"):
        self.probability = probability
        self.grid_ratio = grid_ratio
        self.patch_prob = patch_probabilty
        self.mean = torch.tensor(mean).reshape(-1,1,1)
        self.value = value

    def __call__(self,img:torch.Tensor):
        if random.uniform(0,1)>self.probability:
            return img
        img= img.squeeze()
        c,h,w=torch.tensor(img.shape,dtype=torch.int)
        h_grid_step = torch.round(h*0.25).int()
        w_grid_step = torch.round(w*0.25).int()

        for y in range(0,h,h_grid_step):
            for x in range(0,w,w_grid_step):
                y_end = min(h, y+h_grid_step)  
                x_end = min(w, x+w_grid_step)
                if(random.uniform(0,1) >self.patch_prob):
                    continue
                else:
                    if self.value == 'M':
                        img[:,y:y_end,x:x_end]= self.mean
                    elif self.value == 'R':
                        img[:,y:y_end,x:x_end]=torch.rand_like(img[:,y:y_end,x:x_end])
                    elif self.value =='Z':
                        img[:,y:y_end,x:x_end]=0
                
        return img

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465],value:Literal['R','M','Z']= "Z"):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.value = value
    def __call__(self, img):

        if torch.rand(1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                y1 = random.randint(0, img.size()[1] - h)
                x1 = random.randint(0, img.size()[2] - w)
                if self.value == "M":
                    img[0, y1:y1+h, x1:x1+w] = self.mean[0]
                    img[1, y1:y1+h, x1:x1+w] = self.mean[1]
                    img[2, y1:y1+h, x1:x1+w] = self.mean[2]
                    return img
                elif self.value == "R":
                    img[:, y1:y1+h, x1:x1+w] = torch.rand_like(img[:,y1:y1+h,x1:x1+w])
                    return img
                elif self.value == "Z":
                    img[:, y1:y1+h, x1:x1+w] = 0.0
                    return img
        return img
    

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=1, length=16, probability=1.0):
        self.n_holes = n_holes
        self.length = length
        self.prob = probability
    def __call__(self, img:torch.Tensor):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if random.uniform(0, 1) > self.prob:
            return img
        h = img.size(1)
        w = img.size(2)
        device = img.device
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.0

        mask = torch.from_numpy(mask).to(device=device)
        mask = mask.expand_as(img)
        img = img * mask

        return img

