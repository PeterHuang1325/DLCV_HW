import torch
import argparse
import torch.nn as nn
import torch.optim as optim
#import time
from tqdm import tqdm
#from utils import sample_unlabelled_images
#from model import build_model
#from datasets import get_datasets, get_data_loaders
#from utils import save_model



# Training function.
def train(learner, trainloader, optimizer, device):

    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        images, _ = data
        #images = sample_unlabelled_images()
        images = images.to(device)
        loss = learner(images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        learner.update_moving_average()
        
        train_running_loss += loss.item()
        # Calculate the accuracy.
        #_, preds = torch.max(outputs.data, 1)
        #train_running_correct += (preds == labels).sum().item()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    #epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss

# Validation function.
def validate(model, testloader, criterion, device, pretrained):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            
            # Forward pass.
            outputs = model(image)
                
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    #epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    
    return epoch_loss

