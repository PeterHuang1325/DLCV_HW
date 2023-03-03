import argparse
import datetime
import torch
import wandb
import pandas as pd
import torchvision

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import os 
import imageio


#transform
def get_transform():
    class RescaleChannels(object):
        def __call__(self, sample):
            #img scale between -1 and 1
            return 2 * sample - 1

    return torchvision.transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        RescaleChannels(),
    ])


#MNIST dataset
class HW2_MNIST_Dataset(Dataset):

    def __init__(self, path, info, tfm ,files = None):
        super(HW2_MNIST_Dataset).__init__()
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
        im = np.array(imageio.v2.imread(fname))
        im = self.transform(im)
        label = self.labels[idx]
        return im, label

    
def get_datasets(_dataset_dir):
    train_info = pd.read_csv(os.path.join(_dataset_dir, 'train.csv'))
    val_info = pd.read_csv(os.path.join(_dataset_dir, 'val.csv'))
    train_set = HW2_MNIST_Dataset(os.path.join(_dataset_dir,'data'), info=train_info, tfm=get_transform())
    val_set = HW2_MNIST_Dataset(os.path.join(_dataset_dir,'data'), info=val_info, tfm=get_transform())
    return train_set, val_set   
        

def get_data_loaders(dataset_train, dataset_valid, batch_size, num_workers):
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers
    )
    return train_loader, valid_loader 