"""Dataset setting and data loader for USPS"""

import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import os
import argparse
import torch
import pandas as pd
import torchvision
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import imageio


#USPS dataset
class USPS_Dataset(Dataset):

    def __init__(self, path, info, tfm ,files = None):
        super(USPS_Dataset).__init__()
        self.path = path
        self.info = info
        self.files = [os.path.join(path,x) for x in self.info['image_name']]
        self.labels = [y for y in self.info['label']]
        
        if files != None:
            self.files = files 
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
    
    def __filename__(self,idx):
        return self.files[idx].split('/')[-1]
    
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = np.array(imageio.v2.imread(fname, as_gray=False, pilmode="RGB"))
        im = self.transform(im)
        label = self.labels[idx]
        return im, label

    
def get_datasets(_dataset_dir, tfm):
    train_info = pd.read_csv(os.path.join(_dataset_dir, 'train.csv'))
    val_info = pd.read_csv(os.path.join(_dataset_dir, 'val.csv'))
    train_set = USPS_Dataset(os.path.join(_dataset_dir,'data'), info=train_info, tfm=tfm)
    val_set = USPS_Dataset(os.path.join(_dataset_dir,'data'), info=val_info, tfm=tfm)
    return train_set, val_set   
        

def get_loaders(dataset_train, dataset_valid, batch_size, num_workers):
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(1000)
    np.random.seed(1000)
    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1000)
        
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers
    )
    return train_loader, valid_loader     
    
    
def get_usps(_dataset_dir, batch_size, train):
    """Get USPS datasets loader."""
    # image pre-processing
    tfm = transforms.Compose([transforms.ToPILImage(),
                              transforms.Resize(28),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5)
                                     )])    

    # datasets and data_loader
    train_dataset, val_dataset = get_datasets(_dataset_dir, tfm)
    train_loader, val_loader = get_loaders(train_dataset, val_dataset, batch_size, num_workers=2)
    if train:
        usps_dataloader = train_loader
    else:
        usps_dataloader = val_loader
        
    return usps_dataloader