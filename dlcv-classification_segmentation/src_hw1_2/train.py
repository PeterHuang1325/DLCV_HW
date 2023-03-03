import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
import time
import os
from tqdm.auto import tqdm
from model import build_model
from datasets import HW1Dataset
from utils import *


#training loop
def train_loop(model, model_name, loader, loss_func, optimizer, device):
    model.train()
    train_losses = []
    #train_iou = []
    
    train_out_cut, train_mask_full = [], []
    
    for i, (image, mask) in tqdm(enumerate(loader), total=len(loader)):
        image = image.to(device)
        mask = mask.to(device)
        
        outputs = model(image)
        #out_cut = np.copy(outputs.data.cpu().numpy())
        #out_cut = Variable(out_cut.type(torch.float32), requires_grad=True)
        #loss = loss_func(out_cut, outputs, mask)
        loss = loss_func(outputs['out'], mask.long())
        train_losses.append(loss.item())
        
        out_cut = torch.argmax(outputs['out'], dim=1)
        train_out_cut.append(out_cut.data.cpu().numpy())
        train_mask_full.append(mask.data.cpu().numpy())
        
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    #stack full predicted masks
    train_outs = np.vstack(train_out_cut)
    train_masks = np.vstack(train_mask_full)
    
    #compute IOU
    train_mean_iou = mean_iou_score(train_outs, train_masks)
    train_mean_loss = np.mean(train_losses)
    
    return train_mean_iou, train_mean_loss


#evaluation loop
def eval_loop(model, model_name, loader, loss_func, optimizer, scheduler, device, training=True):
    model.eval()
    val_losses = []
    #val_iou = []
    val_out_cut, val_mask_full = [], []
    
    model.eval()
    with torch.no_grad():
        for step, (image, mask) in tqdm(enumerate(loader), total=len(loader)):
            image = image.to(device)
            mask = mask.to(device)
            
            #out_cut = Variable(out_cut.type(torch.float32), requires_grad=True)
            #loss = loss_func(out_cut, outputs, mask)
            outputs = model(image)
            loss = loss_func(outputs['out'], mask.long())
            val_losses.append(loss.cpu().numpy())
                
            #out_cut = np.copy(outputs.data.cpu().numpy())    
            out_cut = torch.argmax(outputs['out'], dim=1)
            val_out_cut.append(out_cut.data.cpu().numpy())
            val_mask_full.append(mask.data.cpu().numpy())
            
        
        #stack full predicted masks
        val_outs = np.vstack(val_out_cut)
        val_masks = np.vstack(val_mask_full)

        val_mean_iou = mean_iou_score(val_outs, val_masks)
        val_mean_loss = np.mean(val_losses)
        
        #val_mean_iou = np.mean(val_iou)
        if training:
            scheduler.step(val_mean_iou)
        #scheduler.step()
        
    return val_mean_iou, val_mean_loss, val_outs