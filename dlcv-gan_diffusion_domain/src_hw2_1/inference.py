import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
import time
import os
from tqdm import tqdm
import scipy.misc
import random
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.models as models
import torchvision.transforms.functional as TF
import imageio
import glob
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt

# import module
import shutil
from datetime import datetime
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.autograd as autograd
import logging

from trainer import TrainerGAN


def read_options():
    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type=str,
        help='output data directory'
    )
    parser.add_argument(
        '--epochs', type=int, default=150,
        help='Number of epochs to train our network for'
    )
    parser.add_argument(
        '--model_type', type=str, default='DCGAN',
        help='Model for training GAN'
    )
    parser.add_argument(
        '--lr_D', type=float, default=2e-4,
        help='Learning rate for discriminator'
    )
    parser.add_argument(
        '--lr_G', type=float, default=2e-4,
        help='learning rate for generator'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='batch size for training'
    )
    parser.add_argument(
        '--z_dim', type=int, default=100,
        help='z dim for training'
    )
    parser.add_argument(
        '--n_critic', type=int, default=1,
        help='number of iters for training one generator per discriminator'
    )
    parser.add_argument(
        '--num_workers', type=int, default=2,
        help='parellel num workers'
    )
    parser.add_argument(
        '--seed', type=int, default=1000,
        help='seed for experiment'
    )
    parser.add_argument(
        '--clip_value', type=float, default=0.01,
        help='weight clipping for WGAN'
    )
    parser.add_argument(
        '--save_path', type=str, default='./gan_results/',
        help='Save path for GAN training checkpoints'
    )
    parser.add_argument(
        '--workspace_dir', type=str, default='../hw2_data/face/',
        help='Input data for training'
    )
    
    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))
    
    return parsed 


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


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computation device: {device}")
    
    #train GAN
    trainer = TrainerGAN(args)
    trainer.inference('./DCGAN_pass.pth', args['output_path'])
    
    print('INFERENCE COMPLETE')
    print('-'*50)
    
if __name__ == '__main__':
    main()