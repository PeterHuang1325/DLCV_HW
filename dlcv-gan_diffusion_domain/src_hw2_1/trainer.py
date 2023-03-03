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

#call function from other files
from model import *
from dataset import * 


class TrainerGAN():
    def __init__(self, args):
        self.config = args
        
        if self.config['model_type'] == 'DCGAN':
            self.G = DCGAN_Generator(100)
            self.D = DCGAN_Discriminator(3)
        if self.config['model_type'] == 'WGAN-GP':
            self.G = WGAN_GP_Generator(100)
            self.D = WGAN_GP_Discriminator(3)
        
        self.loss = nn.BCELoss() #for DCGAN
        
        """
        NOTE FOR SETTING OPTIMIZER:
        WGAN: use RMSprop optimizer
        WGAN-GP: use Adam optimizer 
        """
        
        if self.config['model_type'] == 'WGAN':
            self.opt_D = torch.optim.RMSprop(self.D.parameters(), lr=self.config["lr_D"])
            self.opt_G = torch.optim.RMSprop(self.G.parameters(), lr=self.config["lr_G"])
        else:
            self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config["lr_D"], betas=(0.5, 0.999))
            self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config["lr_G"], betas=(0.5, 0.999))
            
        self.dataloader = None
        
        self.log_dir = os.path.join(self.config['save_path'], 'logs/')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.ckpt_dir = os.path.join(self.config['save_path'], 'checkpoints/')
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        
        
        FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
        logging.basicConfig(level=logging.INFO, 
                            format=FORMAT,
                            datefmt='%Y-%m-%d %H:%M')
        
        self.steps = 0
        if self.config['model_type'] == 'DCGAN':
            self.z_samples = Variable(torch.randn(100, self.config["z_dim"], 1, 1)).cuda()
        else:
            self.z_samples = Variable(torch.randn(100, self.config["z_dim"])).cuda()
    
    #prepare environment
    def prepare_environment(self):
        """
        Use this funciton to prepare function
        """
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # update dir by time
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join(self.log_dir, time+f'_{self.config["model_type"]}')
        self.ckpt_dir = os.path.join(self.ckpt_dir, time+f'_{self.config["model_type"]}')
        os.makedirs(self.log_dir)
        os.makedirs(self.ckpt_dir)
        
        # create dataset by the above function
        dataset = get_dataset(os.path.join(self.config["workspace_dir"], 'train'))
        self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=2)
        
        # model preparation
        self.G = self.G.cuda()
        self.D = self.D.cuda()
        self.G.train()
        self.D.train()
    
    #gradient penalty
    def gp(self, real_samples, fake_samples):

        """Calculates the gradient penalty loss for WGAN GP"""
        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.D(interpolates)
        fake = Variable(Tensor(real_samples.shape[0]).fill_(1.0), requires_grad=False)
        
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
                      outputs=d_interpolates,
                      inputs=interpolates,
                      grad_outputs=fake,
                      create_graph=True,
                      retain_graph=True,
                      only_inputs=True,
                      )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    #train function
    def train(self):
        """
        Use this function to train generator and discriminator
        """
        self.prepare_environment()
        
        for e, epoch in enumerate(range(self.config["epochs"])):
            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Epoch {e+1}")
            for i, data in enumerate(progress_bar):
                self.D.zero_grad()
                
                imgs = data.cuda()
                bs = imgs.size(0)

                # *********************
                # *    Train D        *
                # *********************
                
                if self.config['model_type'] == 'DCGAN':
                    z = Variable(torch.randn(bs, self.config["z_dim"], 1, 1)).cuda()
                else:
                    z = Variable(torch.randn(bs, self.config["z_dim"])).cuda()
                    
                
                r_imgs = Variable(imgs).cuda()
                
                #DCGAN
                if self.config['model_type'] == 'DCGAN':
                    #train with all real batch
                    r_label = torch.ones((bs)).cuda()
                    r_logit = self.D(r_imgs)
                    r_loss = self.loss(r_logit*0.1, r_label*0.9)
                    r_loss.backward()
                    D_x = r_logit.mean().item()

                    #train with all fake batch
                    f_imgs = self.G(z)
                    f_label = torch.zeros((bs)).cuda()
                    f_logit = self.D(f_imgs)
                    f_loss = self.loss(f_logit, f_label)
                    f_loss.backward()
                    D_G_z1 = f_logit.mean().item()

                    loss_D = r_loss + f_loss
                    self.opt_D.step()
                
                #WGAN-GP
                if self.config['model_type'] == 'WGAN-GP':
                    #real images
                    r_label = torch.ones((bs)).cuda()
                    r_logit = self.D(r_imgs)
                    D_x = r_logit.mean().item()
                    
                    #fake images
                    f_imgs = self.G(z)
                    f_label = torch.zeros((bs)).cuda()
                    f_logit = self.D(f_imgs)
                    
                    gradient_penalty = self.gp(r_imgs, f_imgs)
                    #gradient_penalty = self.gp(r_noised, f_noised)
                    loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + 10*gradient_penalty #lambda=10
                    D_G_z1 = f_logit.mean().item()
                    
                    loss_D.backward()
                    self.opt_D.step()
                    
                '''
                setting loss function
                
                #WGAN
                if self.config['model_type'] == 'WGAN': 
                    loss_D = -torch.mean(r_logit) + torch.mean(f_logit) 
                '''
                
                """ 
                weight clipping
                """                
                if self.config['model_type'] == 'WGAN':
                    #weight clipping for WGAN
                    for p in self.D.parameters():
                        p.data.clamp_(-self.config["clip_value"], self.config["clip_value"])
                
                # *********************
                # *    Train G        *
                # *********************
                if self.steps % self.config["n_critic"] == 0:
                    self.G.zero_grad()
                    
                    # Generate some fake images.
                    if self.config['model_type'] == 'DCGAN':
                        z = Variable(torch.randn(bs, self.config["z_dim"], 1, 1)).cuda()
                    else:
                        z = Variable(torch.randn(bs, self.config["z_dim"])).cuda()
                    
                    f_imgs = self.G(z)
                    # Generator forwarding
                    f_logit = self.D(f_imgs)
                    
                    """
                    NOTE FOR SETTING LOSS FOR GENERATOR:
                    """
                    # Loss for the generator.
                    if self.config['model_type'] == 'DCGAN':
                        #DCGAN with label smoothing
                        loss_G = self.loss(f_logit, r_label)
                    else:
                        loss_G = -torch.mean(self.D(f_imgs)) #WGAN, WGAN-gp

                    # Generator backwarding
                    loss_G.backward()
                    D_G_z2 = f_logit.mean().item()
                    self.opt_G.step()
                    
                if self.steps % 10 == 0:
                    progress_bar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item(), D_x=D_x, D_G_z1=D_G_z1, D_G_z2=D_G_z2)
                self.steps += 1

            self.G.eval()
            f_imgs_sample = ((self.G(self.z_samples).data + 1) / 2.0).clamp(0, 1)
            filename = os.path.join(self.log_dir, f'Epoch_{epoch+1:03d}.png')
            torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
            logging.info(f'Save some samples to {filename}.')

            self.G.train()

            #if (e+1) % 5 == 0 or e == 0:
            if e > 5:
                # Save the checkpoints.
                torch.save(self.G.state_dict(), os.path.join(self.ckpt_dir, f'G_{e}.pth'))
                torch.save(self.D.state_dict(), os.path.join(self.ckpt_dir, f'D_{e}.pth'))

        logging.info('Finish training')

    def inference(self, G_path, outpath, n_generate=1000):
        """
        1. G_path is the path for Generator ckpt
        2. You can use this function to generate final answer
        """

        self.G.load_state_dict(torch.load(G_path))
        self.G.cuda()
        self.G.eval()
        
        #set seed
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
    
        if self.config['model_type'] == 'DCGAN':
            z = Variable(torch.randn(n_generate, self.config["z_dim"], 1, 1)).cuda()
        else:
            z = Variable(torch.randn(n_generate, self.config["z_dim"])).cuda()
        imgs = ((self.G(z).data + 1) / 2.0).clamp(0, 1)
        
        
        os.makedirs(outpath, exist_ok=True)
        '''
        save_dir =  os.path.join(outpath, self.config['model_type'])
        if not os.path.exists(save_dir):
            print(save_dir)
            os.makedirs(save_dir)
        '''
        
        for i in range(n_generate):
            torchvision.utils.save_image(imgs[i], os.path.join(outpath, f'{i+1}.png'))