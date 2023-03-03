import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import os
from tqdm import tqdm
import random
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.models as models
import torchvision.transforms.functional as TF



def save_model(model, model_name):
    """
    Function to save the trained model to disk.
    """
    save_dir = '../outputs_2'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    torch.save({'model_state_dict': model.state_dict()}, f"{save_dir}/model_problem2_{model_name}.pth")
    

    
def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        if (tp_fp+tp_fn-tp) == 0:
            mean_iou += 0
        else:
            iou = tp / (tp_fp + tp_fn - tp)
            mean_iou += iou / 6
        #print('class #%d : %1.5f'%(i, iou))
    #print('\nmean_iou: %f\n' % mean_iou)
    return mean_iou

def dice_coef_loss(pred, labels):
    mean_dice = 0
    smooth = 1
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        dice = (2*tp+smooth) / (tp_fp + tp_fn + smooth)
        mean_dice += dice / 6   
    return 1 - mean_dice

def dice_coeff(pred, labels, reduce_batch_first=False,epsilon=1e-6):
    assert pred.size() == labels.size()
    if pred.dim()==2 or reduce_batch_first:
        inter = torch.dot(pred.reshape(-1), labels.reshape(-1))
        sets_sum = torch.sum(pred) + torch.sum(labels)
        if sets_sum.item() == 0:
            sets_sum = 2* inter
        return (2*inter+epsilon) / (sets_sum + epsilon)
    else:
        dice = 0
        for i in range(pred.shape[0]):
            dice += dice_coeff(pred[i,...],labels[i,...])
        return dice / pred.shape[0]

def multiclass_dice_coeff(pred, labels, reduce_batch_first=False, epsilon=1e-6):
    assert pred.size() == labels.size()
    dice = 0
    for channel in range(pred.shape[1]):
        dice += dice_coeff(pred[:,channel,...], labels[:,channel,...], reduce_batch_first, epsilon)
    return dice / pred.shape[1]

def dice_loss(pred, labels, multiclass=True):
    assert pred.size() == labels.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(pred, labels, reduce_batch_first=True)

def ce_dice_loss(pred, prob, labels, multiclass=True):
    dice = dice_loss(pred.type(torch.float32), labels.type(torch.float32), multiclass=True)
    ce_loss = nn.CrossEntropyLoss()(prob, labels)
    return dice + ce_loss