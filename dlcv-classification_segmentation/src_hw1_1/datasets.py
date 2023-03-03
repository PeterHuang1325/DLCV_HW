import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import os 
import imageio

# Required constants.
_dataset_dir = '../hw1_data/hw1_data/p1_data/'

# Training transforms
def get_train_transform(image_size, pretrained):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return train_transform

# Validation transforms
def get_valid_transform(image_size, pretrained):
    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return valid_transform

# Image normalization transforms.
def normalize_transform(pretrained):
    if pretrained: # Normalization for pre-trained weights.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    else: # Normalization when training from scratch.
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    return normalize


class HW1Dataset(Dataset):

    def __init__(self,path, tfm ,files = None):
        super(HW1Dataset).__init__()
        self.path = path
        self.files = [os.path.join(path,x) for x in os.listdir(path) if x.endswith(".png")]
        #self.files = random.sample(self.files, len(self.files))
        if files != None:
            self.files = files
        #print(f"One {path} sample",self.files[0]) 
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
    
    def __filename__(self,idx):
        return self.files[idx].split('/')[-1]
    
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = np.array(imageio.v2.imread(fname))
        #print(type(im))
        im = self.transform(im)
        #im = self.data[idx]
        label = int(fname.split('/')[-1].split('_')[0])
        return im, label

    
def get_datasets(image_size, pretrained):
    get_valid_transform(image_size, pretrained)
    train_set = HW1Dataset(os.path.join(_dataset_dir,'train_50'), tfm=get_train_transform(image_size, pretrained))
    val_set = HW1Dataset(os.path.join(_dataset_dir,'val_50'), tfm=get_valid_transform(image_size, pretrained))
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