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
        '--source_path', type=str,
        help='source data directory'
    )
    parser.add_argument(
        '--target_path', type=str,
        help='target data directory'
    )
    parser.add_argument(
        '--output_path', type=str,
        help='output features directory'
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


def main():    
    
    args = read_options()
    
    source_name = args['source_path'].split('/')[-1] 
    target_name = args['target_path'].split('/')[-1] 
    feat_path = args['output_path']
    print(source_name, target_name)
    
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
    src_data_loader_eval = get_data_loader(source_name, args['source_path'], args['batch_size'], train=False)
    tgt_data_loader_eval = get_data_loader(target_name, args['target_path'], args['batch_size'], train=False)
    
    #set storage
    embed_list, class_labels, domain_labels  = [], [], []
    #set model
    model = init_model(net=MNISTmodel(), restore=None)

    #load model
    if target_name == 'svhn':
        model.load_state_dict(torch.load('./mnistm-svhn-dann.pt'))
    if target_name == 'usps':
        model.load_state_dict(torch.load('./mnistm-usps-dann.pt'))
    
    model.eval()
    #source eval data
    for (images, labels) in tqdm(src_data_loader_eval):
        images = images.to(device)
        labels = labels.to(device)
        size = len(labels)
        #domain labels
        domain = torch.zeros(size).long().to(device)
        domain_labels.append(domain.data.cpu().numpy())
        #get features
        feature = model.feature(images)
        embed_list.append(feature.data.cpu().numpy())
        #get labels
        class_labels.append(labels.data.cpu().numpy())
        
    #target eval data
    for (images, labels) in tqdm(tgt_data_loader_eval):
        images = images.to(device)
        labels = labels.to(device)
        size = len(labels)
        
        #preds, domain = model(images, alpha=0)
        #pred_class = preds.data.max(1)[1]
        #pred_domain = domain.data.max(1)[1]
        
        domain = torch.ones(size).long().to(device)
        #domain labels
        domain_labels.append(domain.cpu().numpy())
        
        #get features
        feature = model.feature(images)
        embed_list.append(feature.data.cpu().numpy())
        
        #get labels
        class_labels.append(labels.data.cpu().numpy())
    
    
    class_total = np.hstack(class_labels)
    domain_total = np.hstack(domain_labels)
    #embeds = np.array(embed_list)
    embeds = np.vstack(embed_list)
    embeds = embeds.reshape(embeds.shape[0], embeds.shape[1]*embeds.shape[2]*embeds.shape[3])
    print(embeds.shape)

    #t-SNE
    print('Perform t-SNE:')
    #print(len(feat_class_list), np.vstack(feat_class_list[-1]).shape)
    print(f'target length: {len(tgt_data_loader_eval)}', f'source length: {len(src_data_loader_eval)}')
    tsne_feat = get_tsne(embeds)
    
    if not os.path.exists(feat_path):
        os.makedirs(feat_path)
    #save tsne and label results   
    np.savez(os.path.join(feat_path, f'tsne_info_{target_name}.npz'), 
             tsne_feat=tsne_feat, lbl_class=class_total, lbl_domain=domain_total)

if __name__ == '__main__':
    main()
