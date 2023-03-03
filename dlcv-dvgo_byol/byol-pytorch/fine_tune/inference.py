# Import necessary packages.
import torch
import numpy as np
import argparse
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset, Dataset
import imageio
import time
import os
from tqdm.auto import tqdm
from model import build_model
from utils import focal_loss
import pandas as pd
from lbl_enc import enc2class 


def read_options():
    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv_path', type=str,
        help='input csv file directory'
    )
    parser.add_argument(
        '--input_path', type=str,
        help='input data directory'
    )
    parser.add_argument(
         '--output_path', type=str,
        help='output data directory'
    )
    parser.add_argument(
        '--image_size', type=int, default=128,
        help='image size for training the model'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128, 
        help='batch size for training'
    )
    parser.add_argument(
        '--num_classes', type=int, default=65,
        help='number of label classes'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='parellel num workers'
    )
    parser.add_argument(
        '--seed', type=int, default=1000,
        help='seed for experiment'
    )

    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))
    
    return parsed 

#transform
def get_eval_transform(image_size):
    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])
    return eval_transform

#dataset
class HW4Dataset(Dataset):

    def __init__(self, df, path, tfm, files = None):
        super(HW4Dataset).__init__()
        #self.path = path
        self.df = df
        self.files = [os.path.join(path, filename) for filename in self.df['filename']]

        if files != None:
            self.files = files
        #print(f"One {path} sample",self.files[0]) 
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
    
    def __filename__(self,idx):
        return os.path.split(self.files[idx])[-1]
    
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = np.array(imageio.v2.imread(fname))
        #print(type(im))
        im = self.transform(im)
        return im  
  
    
#main function
def main():
    
    #read parsed
    args = read_options()
    
    #set seed
    myseed = args['seed']  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
    
    # Load the training and validation datasets.
    eval_df = pd.read_csv(args['csv_path'])
    eval_set = HW4Dataset(eval_df, args['input_path'], tfm=get_eval_transform(args['image_size']))
    print(f"[INFO]: Number of validation images: {len(eval_set)}")

    # Load the training and validation data loaders.
    eval_loader = DataLoader(eval_set, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Computation device: {device}")
    
    #build model
    model = models.resnet50(weights=None).to(device)
    model.load_state_dict(torch.load('./byol-pytorch/fine_tune/resnet50_finetuned_True_C.pth'))
    
    model.eval()
    prediction = []

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            # image and labels
            imgs = batch
            eval_pred = model(imgs.to(device))
            eval_label = np.argmax(eval_pred.cpu().data.numpy(), axis=1)
            class_lbl = [enc2class[enc] for enc in eval_label]
            prediction += class_lbl
            #prediction += eval_label.squeeze().tolist()
    
    #get files
    eval_files = [eval_set.__filename__(idx) for idx in range(len(eval_set))]
    ids = list(range(0, len(eval_set), 1))
    
    out_df = pd.DataFrame(np.array([ids, eval_files, prediction], dtype=object).T, columns=['id', 'filename', 'label'])
    print('EVALUATION COMPLETE')
    print('-'*50)
    
    save_path = args['output_path']
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    '''
    #output to csv
    out_df.to_csv(os.path.join(save_path), index=False)
    
if __name__ == '__main__':
    main()
