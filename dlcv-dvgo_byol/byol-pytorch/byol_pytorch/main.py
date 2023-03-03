import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
import time
import os
from tqdm.auto import tqdm
from model import build_model
from datasets import get_datasets, get_data_loaders
from utils import save_model, focal_loss
from train import train, validate
from byol_pytorch import BYOL

def read_options():
    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs', type=int, default=500,
        help='Number of epochs to train our network for'
    ) #50
    parser.add_argument(
        '--pretrained', action='store_true', default=False,
        help='Whether to use pretrained weights or not'
    )
    parser.add_argument(
        '--learning_rate', type=float,
        dest='learning_rate', default=3e-4, #5e-4
        help='Learning rate for training the model'
    )
    parser.add_argument(
        '--image_size', type=int, default=128,
        help='image size for training the model'
    )
    parser.add_argument(
        '--batch_size', type=int, default=256, 
        help='batch size for training'
    )
    parser.add_argument(
        '--num_classes', type=int, default=64,
        help='number of label classes'
    )
    parser.add_argument(
        '-nw', '--num_workers', type=int, default=4,
        help='parellel num workers'
    )
    parser.add_argument(
        '-sd', '--seed', type=int, default=1000,
        help='seed for experiment'
    )
    parser.add_argument(
        '-pa', '--patience', type=int, default=20,
        help='patience for early stop'
    )

    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))
    
    return parsed 

#focal loss
def focal_loss(dataset):
    #focal loss
    focal_loss = torch.hub.load(
                'adeelh/pytorch-multi-class-focal-loss',
                model='FocalLoss',
                #alpha=weights.to(device),
                gamma=2,
                reduction='mean',
                force_reload=False)
    return focal_loss

#main function
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
    
    # Load the training and validation datasets.
    train_set = get_datasets(args['image_size'], args['pretrained'])
    dataset_classes = args['num_classes']
    
    print(f"[INFO]: Number of training images: {len(train_set)}")

    # Load the training and validation data loaders.
    train_loader = get_data_loaders(train_set, args['batch_size'], args['num_workers'])
    # Learning_parameters. 
    lr = args['learning_rate']
    epochs = args['epochs']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")
    #build model
    model = build_model(
        pretrained=args['pretrained'], 
        fine_tune=True, 
        num_classes=args['num_classes']
    ).to(device)
    
    #BYOL
    learner = BYOL(model, image_size = args['image_size'], hidden_layer = 'avgpool')
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    
    # Optimizer.
    #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    optimizer = torch.optim.Adam(learner.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args['epochs'], last_epoch=-1)
    
    # Loss function.
    #criterion = nn.CrossEntropyLoss()
    #criterion = focal_loss(train_set) #nn.CrossEntropyLoss()
    # Lists to keep track of losses and accuracies.
    train_loss = []
    
    best_loss = np.inf
    stale = 0 
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss = train(learner, train_loader, optimizer, device)
        
        train_loss.append(train_epoch_loss)
        scheduler.step()
        
        save_model(epoch, model, args['pretrained'])
        if train_epoch_loss < best_loss:
            best_loss = train_epoch_loss
            
            # Save the trained model weights.
            print(f'Save best model at {epoch+1} epoch:')
            #save_model(epochs, model, args['pretrained'])
            stale = 0
        else:
            stale += 1
            if stale > args['patience']:
                print(f"No improvment {args['patience']} consecutive epochs, early stopping")
                break
            
        print(f"Training loss: {train_epoch_loss:.3f}")
        print('-'*50)
        time.sleep(5)
        
        
    print('TRAINING COMPLETE')
    print('-'*50)
    
    
if __name__ == '__main__':
    main()