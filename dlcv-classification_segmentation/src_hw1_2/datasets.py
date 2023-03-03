import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import os
from tqdm import tqdm
import random
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.models as models
import torchvision.transforms.functional as TF
import imageio
from PIL import Image
import PIL.ImageOps 


'''
class HW1Dataset(Dataset):
    def __init__(self, filepath):
        self.mask_file = sorted([os.path.join(filepath, file) for file in os.listdir(filepath) if file.endswith('.png')])
        self.imgs_file = sorted([os.path.join(filepath, file) for file in os.listdir(filepath) if file.endswith('.jpg')])
        #self.img_size = image_size
        
    def __len__(self):
        return len(self.imgs_file)
    
    def transform(self, image, mask):
        
        #random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(image.size()[1], image.size()[2]))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        
        #color jitter
        image = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,hue=0.5)(image)
        
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        return image, mask
    
    def __getitem__(self, idx):
        
        masks = torch.empty((self.__len__(), 512, 512))
        #masks
        mask = imageio.v2.imread(self.mask_file[idx])
        mask = (mask >= 128).astype(int)
        
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        
        masks[idx, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[idx, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[idx, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[idx, mask == 2] = 3  # (Green: 010) Forest land 
        masks[idx, mask == 1] = 4  # (Blue: 001) Water 
        masks[idx, mask == 7] = 5  # (White: 111) Barren land 
        masks[idx, mask == 0] = 6  # (Black: 000) Unknown
        
        mask = masks[idx]
        #images
        image = torch.from_numpy(imageio.v2.imread(self.imgs_file[idx])/255).permute(2,0,1).type(torch.float32)
        image, mask = self.transform(image=image, mask=mask)
        
        return image, mask


'''

class HW1Dataset(Dataset):
    def __init__(self, filepath):
        self.mask_file = sorted([os.path.join(filepath, file) for file in os.listdir(filepath) if file.endswith('.png')])
        self.imgs_file = sorted([os.path.join(filepath, file) for file in os.listdir(filepath) if file.endswith('.jpg')])
        
    def __len__(self):
        return len(self.imgs_file)
  
    def __filename__(self, idx):
        return self.imgs_file[idx].split('/')[-1].split('_')[0]
    
    def __getitem__(self, idx):
        
        masks = torch.empty((self.__len__(), 512, 512))
        #masks
        mask = imageio.v2.imread(self.mask_file[idx])
        mask = (mask >= 128).astype(int)
        
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        
        masks[idx, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[idx, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[idx, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[idx, mask == 2] = 3  # (Green: 010) Forest land 
        masks[idx, mask == 1] = 4  # (Blue: 001) Water 
        masks[idx, mask == 7] = 5  # (White: 111) Barren land 
        masks[idx, mask == 0] = 6  # (Black: 000) Unknown
        
        mask = masks[idx]
        #mask = mask.astype(np.uint8) #for transform to PIL
        #images
        image = torch.from_numpy(imageio.v2.imread(self.imgs_file[idx])/255).permute(2,0,1).type(torch.float32)
        #data augmentation
        #image, mask = self.transform(image=image, mask=mask)
        image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
        #mask = mask*255
        
        return image, mask
