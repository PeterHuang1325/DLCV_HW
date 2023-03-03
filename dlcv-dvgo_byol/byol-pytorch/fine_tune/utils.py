import torch
import numpy as np
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

def save_model(epoch, model, optimizer, criterion, pretrained):
    """
    Function to save the trained model to disk.
    """
    save_dir = './outputs'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    torch.save(model.state_dict(), f"{save_dir}/resnet50_finetuned_{pretrained}.pth")
