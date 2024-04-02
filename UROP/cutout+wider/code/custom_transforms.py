from __future__ import absolute_import
from typing import Any,List,Literal

from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch
import matplotlib.patches as patches
import torchvision

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=1, length=100, probability=1.0):
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
        totensor=torchvision.transforms.ToTensor()
        tensor_pil=totensor(img)
        
        if random.uniform(0, 1) > self.prob:
            return img
        h = tensor_pil.shape[1]
        w = tensor_pil.shape[2]
        self.length=h//4
        device = tensor_pil.device
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
        mask = mask.expand_as(tensor_pil)
        tensor_pil = tensor_pil * mask

        return tensor_pil