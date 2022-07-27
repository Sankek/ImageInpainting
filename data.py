import numpy as np
import os

import torch
from torchvision import transforms as tt
from torch.utils.data import Dataset
from PIL import Image

from masks import MaskGenerator


class DataFolder(Dataset):
    def __init__(self, path, dataset_mean=[0.5]*3, dataset_std=[0.5]*3, ext='.jpg'):
        super().__init__()
        self.path = path
        self.ext = ext
        self.mean = dataset_mean
        self.std = dataset_std

        self.files = list(filter(self.name_filter, os.listdir(self.path)))
        self.len_ = len(self.files)

        self.transform = tt.Compose([
            tt.ToTensor(),
            tt.Normalize(self.mean, self.std) 
        ])
        
    
    def name_filter(self, name):
        if not name.endswith('.' + self.ext):
            return False
        
        return True

        
    def __len__(self):
        return self.len_

    
    def __getitem__(self, index):
        file = os.path.join(self.path, self.files[index])
        img = Image.open(file)
        
        img = self.transform(img)
        
        return img
    
    
class InpaintingDataset(DataFolder):
    def __init__(self, path, image_size, channels=3, dataset_mean=[0.5]*3, dataset_std=[0.5]*3, ext='.jpg'):
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
    
    
class InpaintingPngMaskDataset(DataFolder):
    def __init__(self, path, image_size, channels=3, dataset_mean=[0.5]*3, dataset_std=[0.5]*3, ext='jpg', mask_suffix='_mask.png'):
        super().__init__(path, dataset_mean=dataset_mean, dataset_std=dataset_std, ext=ext)
        
        self.mask_suffix = mask_suffix
        self.image_size = image_size
        self.channels = channels
        
        
    def name_filter(self, name):
        if not name.endswith('.' + self.ext):
            return False
        if name.endswith(self.mask_suffix):
            return False
        
        return True

        
    def __getitem__(self, index):
        name = self.files[index][:-len('.' + self.ext)]
        mask_name = name + self.mask_suffix
        
        file = os.path.join(self.path, self.files[index])
        mask_file = os.path.join(self.path, mask_name)
        
        img = Image.open(file)
        img = self.transform(img)
        
        mask = Image.open(mask_file)
        mask = np.array(mask)[:, :, 3]
        valid = mask == 255
        hole = ~valid
        mask[valid] = 255
        mask[hole] = 0
        mask = torch.FloatTensor(mask/255)
        mask = mask.unsqueeze(0).expand(self.channels, -1, -1)
        
        return img*mask, mask, img
    