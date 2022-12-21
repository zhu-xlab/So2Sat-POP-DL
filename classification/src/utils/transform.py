# Data Augmentations

import os
import torchvision.transforms.functional as TF
import random
import torch
from utils.constants import config_path
from utils.file_folder_ops import load_json

from utils.utils import *


class RandomRotationTransform(torch.nn.Module):
    """Rotate by one of the given angles.
    Args:
        angles (sequence): sequence of rotation angles
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, angles, p=0.5):
        self.angles = angles
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            angle = random.choice(self.angles)
            return TF.rotate(x, angle)
        return x


class RandomGamma(torch.nn.Module):
    """Perform gamma correction on an image.

    Also known as Power Law Transform.
    """
    def __init__(self, gamma_limit=(0.8, 1.2), p=0.5):
        self.gamma_limit = gamma_limit
        self.p = p
        dataset_stats = load_json(os.path.join(config_path, 'dataset_stats', 'mod_dataset_stats.json'))
        self.data_mean = dataset_stats['sen2spring']['mean']
        self.data_std = dataset_stats['sen2spring']['std']
        self.data_max = dataset_stats['sen2spring']['max']

    def __call__(self, x):
        if torch.rand(1) < self.p:
            gamma = random.uniform(self.gamma_limit[0], self.gamma_limit[1])
            for i in range(3):
                x[i:i+1] = x[i:i+1] * self.data_std[i]
                x[i:i+1] = x[i:i+1] + self.data_mean[i]
                x[i:i+1] = x[i:i+1] / self.data_max[i]
            x[:3] = torch.clip(x[:3], min=0)
            x[:3] = TF.adjust_gamma(x[:3], gamma)
            for i in range(3):
                x[i:i+1] = x[i:i+1] * self.data_max[i]
                x[i:i+1] = x[i:i+1] - self.data_mean[i]
                x[i:i+1] = x[i:i+1] / self.data_std[i]
        return x


class RandomBrightness(torch.nn.Module):
    """Adds random brightness on an image.
    """
    def __init__(self, beta_limit=(0.8, 1.2), p=0.5):
        self.beta_limit = beta_limit
        self.p = p
        dataset_stats = load_json(os.path.join(config_path, 'dataset_stats', 'mod_dataset_stats.json'))
        self.data_mean = dataset_stats['sen2spring']['mean']
        self.data_std = dataset_stats['sen2spring']['std']
        self.data_max = dataset_stats['sen2spring']['max']

    def __call__(self, x):
        if torch.rand(1) < self.p:
            beta = random.uniform(self.beta_limit[0], self.beta_limit[1])
            for i in range(3):
                x[i:i+1] = x[i:i+1] * self.data_std[i]
                x[i:i+1] = x[i:i+1] + self.data_mean[i]
                x[i:i+1] = x[i:i+1] / self.data_max[i]
            x[:3] = TF.adjust_brightness(x[:3], beta)
            for i in range(3):
                x[i:i+1] = x[i:i+1] * self.data_max[i]
                x[i:i+1] = x[i:i+1] - self.data_mean[i]
                x[i:i+1] = x[i:i+1] / self.data_std[i]
        return x
