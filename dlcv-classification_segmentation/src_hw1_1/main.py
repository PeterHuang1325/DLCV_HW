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
from utils import save_model, focal_loss, get_pca, get_tsne
from train import train, validate

def read_options():
    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--epochs', type=int, default=50,
        help='Number of epochs to train our network for'
    )
    parser.add_argument(
        '-pt', '--pretrained', action='store_true', default=False,
        help='Whether to use pretrained weights or not'
    )
    parser.add_argument(
        '-lr', '--learning_rate', type=float,
        dest='learning_rate', default=1e-4,
        help='Learning rate for training the model'
    )
    parser.add_argument(
        '-im', '--image_size', type=int, default=128,
        help='image size for training the model'
    )
    parser.add_argument(
        '-bs', '--batch_size', type=int, default=125,
        help='batch size for training'
    )
    parser.add_argument(
        '-nc', '--num_classes', type=int, default=50,
        help='number of label classes'
    )
    parser.add_argument(
        '-nw', '--num_workers', type=int, default=0,
        help='parellel num workers'
    )
    parser.add_argument(
        '-sd', '--seed', type=int, default=1000,
        help='seed for experiment'
    )
    parser.add_argument(
        '-pa', '--patience', type=int, default=10,
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
    train_set, valid_set = get_datasets(args['image_size'], args['pretrained'])
    dataset_classes = args['num_classes']
    
    print(f"[INFO]: Number of training images: {len(train_set)}")
    print(f"[INFO]: Number of validation images: {len(valid_set)}")

    # Load the training and validation data loaders.
    train_loader, valid_loader = get_data_loaders(train_set, valid_set, args['batch_size'], args['num_workers'])
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
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    
    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args['epochs'], last_epoch=-1)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)
    # Loss function.
    #criterion = nn.CrossEntropyLoss()
    criterion = focal_loss(train_set) #nn.CrossEntropyLoss()
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    
    best_acc = 0
    stale = 0 
    feat_list = []
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion, device)
        valid_epoch_loss, valid_epoch_acc, feat_full = validate(model, valid_loader, criterion, device, args['pretrained'])
        
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        scheduler.step()
        
        if not args['pretrained']:
            #append second last layer
            feat = np.vstack(feat_full)
            feat_list.append(feat) #features['fc1'].cpu().numpy()
        
        if valid_epoch_acc > best_acc:
            best_acc = valid_epoch_acc
            
            # Save the trained model weights.
            print(f'Save best model at {epoch+1} epoch:')
            save_model(epochs, model, optimizer, criterion, args['pretrained'])
            stale = 0
        else:
            stale += 1
            if stale > args['patience']:
                print(f"No improvment {args['patience']} consecutive epochs, early stopping")
                break
            
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)
        time.sleep(5)
        
        
    print('TRAINING COMPLETE')
    print('-'*50)
    save_path = '../outputs/layers'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    #save threshold
    if args['pretrained'] is False:
        #print('length:', len(feat_list))
        #print('arr shape:',feat_list[-1].shape)
        
        #get label
        val_lbl = np.array([valid_set.__getitem__(idx)[1] for idx in range(len(valid_set))])
        #split_idx = int(len(valid_set) / args['batch_size']) #split idx
        #print('split_idx:', split_idx)
        
        #PCA
        print('Perform PCA:')
        pca_result = get_pca(feat_list[-1])
        np.savez(os.path.join(save_path,'pca_info.npz'), arr = pca_result, lbl = val_lbl)
        
        #t-SNE
        print('Perform t-SNE:')
        tsne_1 = get_tsne(feat_list[0]) #first
        tsne_2 = get_tsne(feat_list[int(round(0.5*len(feat_list)))]) #middle
        tsne_3 = get_tsne(feat_list[-1]) #last
        np.savez(os.path.join(save_path,'tsne_info.npz'), 
                 tsne_1=tsne_1, tsne_2=tsne_2,
                 tsne_3=tsne_3, lbl=val_lbl)
    
if __name__ == '__main__':
    main()