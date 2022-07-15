import numpy as np
import os

import torch
from torchvision import transforms as tt
from torch.utils.data import Dataset
from PIL import Image

from masks import MaskGenerator


class DataFolder(Dataset):
    def __init__(self, path, dataset_mean=[0.5]*3, dataset_std=[0.5]*3, ext='jpg'):
        super().__init__()
        self.path = path
        self.ext = ext
        self.mean = dataset_mean
        self.std = dataset_std

        self.files = list(filter(lambda f: f.endswith(self.ext), os.listdir(self.path)))
        self.len_ = len(self.files)

        self.transform = tt.Compose([
            tt.ToTensor(),
            tt.Normalize(self.mean, self.std) 
        ])

        
    def __len__(self):
        return self.len_

    
    def __getitem__(self, index):
        file = os.path.join(self.path, self.files[index])
        img = Image.open(file)
        
        img = self.transform(img)
        
        return img
    
    
class InpaintingDataset(DataFolder):
    def __init__(self, path, image_size, channels=3, dataset_mean=[0.5]*3, dataset_std=[0.5]*3, ext='jpg'):
        super().__init__(path, dataset_mean=dataset_mean, dataset_std=dataset_std, ext=ext)
    
        self.image_size = image_size
        self.channels = channels
        self.mask_generator = MaskGenerator(image_size, image_size)

        
    def __getitem__(self, index):
        file = os.path.join(self.path, self.files[index])
        img = Image.open(file)
        
        img = self.transform(img)
        mask = torch.FloatTensor(self.mask_generator.generate()).unsqueeze(0).expand(self.channels, -1, -1)
        
        return img*mask, mask, img