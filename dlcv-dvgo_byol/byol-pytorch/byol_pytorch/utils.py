import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os


#focal loss
def focal_loss(Dataset):
    #focal loss
    focal_loss = torch.hub.load(
                'adeelh/pytorch-multi-class-focal-loss',
                model='FocalLoss',
                #alpha=weights.to(device),
                gamma=2,
                reduction='mean',
                force_reload=False)
    return focal_loss

'''
def sample_unlabelled_images():
    return torch.randn(20, 3, 128, 128)
'''
def save_model(epoch, model, pretrained):
    """
    Function to save the trained model to disk.
    """
    save_dir = './outputs'
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    '''
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir}/resnet50_pretrained_{pretrained}_{epoch}.pth")