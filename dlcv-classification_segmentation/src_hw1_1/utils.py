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

def save_model(epochs, model, optimizer, criterion, pretrained):
    """
    Function to save the trained model to disk.
    """
    save_dir = '../outputs'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    torch.save({'model_state_dict': model.state_dict()}, f"{save_dir}/model_pretrained_{pretrained}.pth")
    

def get_pca(hidden):
    #data_arr = np.vstack(hidden)
    #do pca
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(hidden) #data_arr
    return pca_result    
    
def get_tsne(hidden):
    #data_arr = np.vstack(hidden)
    X_tsne = TSNE(n_components=2, init='random', random_state=1000, verbose=1).fit_transform(hidden)
    # Normalization the processed features 
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    return X_norm