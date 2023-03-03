import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import os 
import imageio
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# Required constants.
_dataset_dir = '/workspace/hw4-PeterHuang1325/hw4_data/mini/'

def split_data(path, df):
    train_df, val_df = train_test_split(df, stratify=df['label'], test_size=0.2, random_state=1000)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df

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
        '''
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        '''
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    return normalize


class HW4Dataset(Dataset):

    def __init__(self, df, tfm ,files = None):
        super(HW4Dataset).__init__()
        #self.path = path
        self.df = df
        self.files = [os.path.join(_dataset_dir,'train',filename) for filename in self.df['filename']]
        self.labels = list(self.df['label'])

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
        label = self.labels[idx]
        return im, label

    
def get_datasets(image_size, pretrained):
    files_df = pd.read_csv(_dataset_dir+'train.csv')
    '''
    #cope with the label with labelencoder
    labelencoder = LabelEncoder()
    files_df['label'] = labelencoder.fit_transform(files_df['label'])
    #split data
    train_df, val_df = split_data(_dataset_dir, files_df)
    '''
    data_set = HW4Dataset(files_df, tfm=get_train_transform(image_size, pretrained))
    return data_set   
        

def get_data_loaders(dataset, batch_size, num_workers):
    data_loader = DataLoader(
        dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers
    )
    return data_loader