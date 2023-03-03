import torch.utils.data
import torch.nn as nn
import numpy as np
import os
from utils.utils import get_data_loader, get_tsne, init_model, init_random_seed
from models.model import MNISTmodel
from tqdm import tqdm
import torch.utils.data as data
from PIL import Image
import argparse
import torch
import pandas as pd
import random
import torchvision

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import imageio


def read_options():
    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path', type=str,
        help='input data directory'
    )
    parser.add_argument(
        '--output_path', type=str,
        help='output data directory'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='batch size for training'
    )
    parser.add_argument(
        '--num_workers', type=int, default=2,
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


# image pre-processing
tfm = transforms.Compose([transforms.ToPILImage(),
                          transforms.Resize(28),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5)
                                 )])    

#Infer_Dataset
class Infer_Dataset(Dataset):

    def __init__(self, path, name, tfm ,files = None):
        super(Infer_Dataset).__init__()
        self.path = path
        #self.info = info
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path)])
        self.name = name
        #self.labels = [y for y in self.info['label']]
        
        if files != None:
            self.files = files 
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
    
    def __filename__(self,idx):
        split_names = self.files[idx].split('/')
        return self.files[idx].split('/')[-1]
    
    def __getitem__(self,idx):
        fname = self.files[idx]
        if self.name == 'usps':
            im = np.array(imageio.v2.imread(fname, as_gray=False, pilmode="RGB"))
        if self.name == 'svhn':
            im = np.array(imageio.v2.imread(fname))
        im = self.transform(im)
        #label = self.labels[idx]
        return im #, label

def main():    
    
    args = read_options()
    if 'svhn' in args['input_path']:
        tgt_dataset = 'svhn'
    if 'usps' in args['input_path']:
        tgt_dataset = 'usps'
        
    #tgt_dataset = args['input_path'].split('/')[-2] 
    #tgt_image_root = os.path.join(args['input_path'], tgt_dataset)
    
    #set seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])
                                   
    #set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device used: {device}')
    
    # load dataset
    test_set = Infer_Dataset(args['input_path'], name=tgt_dataset, tfm=tfm)
    #print('length dataset:', len(test_set))
    test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'])
    
    #build model
    model = init_model(net=MNISTmodel(), restore=None)
    #load model
    if tgt_dataset == 'svhn':
        model.load_state_dict(torch.load('./src_hw2_3/mnistm-svhn-dann.pt'))
    if tgt_dataset == 'usps':
        model.load_state_dict(torch.load('./src_hw2_3/mnistm-usps-dann.pt'))
    
    """Evaluate model for dataset."""
    # set eval state for Dropout and BN layers
    model.eval()
    
    pred_list = []
    # evaluate network
    for images in tqdm(test_loader):
        images = images.to(device)
        preds, domain = model(images, alpha=0)
        
        pred_cls = preds.data.max(1)[1]
        pred_domain = domain.data.max(1)[1]
        pred_list += pred_cls.cpu().numpy().tolist()
        
    #get files
    test_files = [test_set.__filename__(idx) for idx in range(len(test_set))]
    out_df = pd.DataFrame(np.array([test_files, pred_list], dtype=object).T, columns=['image_name', 'label'])
    print('EVALUATION COMPLETE')
    print('-'*50)
    
    save_path = args['output_path']
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    '''
    #output to csv
    #os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    #out_df.to_csv(os.path.join(os.path.split(save_path)[0], os.path.split(save_path)[1]), index=False)
    out_df.to_csv(os.path.join(save_path), index=False)
if __name__ == '__main__':
    main()

