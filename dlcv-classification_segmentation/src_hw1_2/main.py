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
from train import train_loop, eval_loop

def read_options():
    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--epochs', type=int, default=60,
        help='Number of epochs to train our network'
    )
    parser.add_argument(
        '-md', '--model', type=str, default='VGG16_FCN32',
        help='model used for training'
    )
    parser.add_argument(
        '-lr', '--learning_rate', type=float,
        dest='learning_rate', default=1e-4,
        help='Learning rate for training the model'
    )
    parser.add_argument(
        '-im', '--image_size', type=int, default=512,
        help='image size for training the model'
    )
    parser.add_argument(
        '-bs', '--batch_size', type=int, default=20,
        help='batch size for training'
    )
    parser.add_argument(
        '-nc', '--num_classes', type=int, default=7,
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
    path = '../hw1_data/hw1_data/p2_data/'
    train_set = HW1Dataset(os.path.join(path,'train/'))
    valid_set = HW1Dataset(os.path.join(path,'validation/'))
    
    print(f"[INFO]: Number of training images: {len(train_set)}")
    print(f"[INFO]: Number of validation images: {len(valid_set)}")

    # Load the training and validation data loaders.
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'], pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'], pin_memory=True)
    # Learning_parameters. 
    lr = args['learning_rate']
    epochs = args['epochs']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")
    #build model
    model = build_model(args['model'],num_classes=args['num_classes']).to(device)
        
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    
    # Optimizer.
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args['epochs'], last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=3)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)
    # Loss function.
    criterion = nn.CrossEntropyLoss()
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    

    # Start the training.
    train_loss_history = []
    train_iou_history = []
    val_loss_history = []
    val_iou_history = []
    val_full_outs = [] #for plotting mask
    best_iou = 0
    stale = 0 
    
    for epoch in range(epochs):
        
        train_mean_iou, train_mean_loss = train_loop(model, args['model'], train_loader, criterion, optimizer, device)
        val_mean_iou, val_mean_loss, val_outs = eval_loop(model, args['model'], valid_loader, criterion, optimizer, scheduler, device)
            
        #train history
        train_iou_history.append(train_mean_iou)
        train_loss_history.append(train_mean_loss)
        
        #validation history
        val_iou_history.append(val_mean_iou)
        val_loss_history.append(val_mean_loss)
        print('Epoch: {}/{} |  Train Loss: {:.3f}, Val Loss: {:.3f}, Train IOU: {:.3f}, Val IOU: {:.3f}'.format(epoch+1, epochs,
                                                                                                                train_mean_loss,
                                                                                                                val_mean_loss,
                                                                                                                train_mean_iou,
                                                                                                                val_mean_iou))
        #store predicted masks
        val_full_outs.append(val_outs) #shape: (257, 1, 512, 512)
        #check best iou
        if val_mean_iou > best_iou:
            best_iou = val_mean_iou
            
            # Save the trained model weights.
            print(f'Save best model at {epoch+1} epoch:')
            save_model(model, args['model'])
            stale = 0
        else:
            stale += 1
            if stale > args['patience']:
                print(f"No improvment {args['patience']} consecutive epochs, early stopping")
                break
            
        print(f"Training loss: {train_mean_loss:.3f}, training iou: {train_mean_iou:.3f}")
        print(f"Validation loss: {val_mean_loss:.3f}, validation iou: {val_mean_iou:.3f}")
        print('-'*50)
        time.sleep(5)
        
        
    print('TRAINING COMPLETE')
    print('-'*50)
    
    #save target pred masks
    save_path = '../outputs_2/preds'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    target_seg = [13, 62, 104]
    mid = int(len(val_full_outs) // 2) #mid idx
    for idx in target_seg:
        np.savez(os.path.join(save_path,f'pred_00{idx}.npz'),
                 first = val_full_outs[0][idx], 
                 middle = val_full_outs[mid][idx],
                 last = val_full_outs[-1][idx])
        
if __name__ == '__main__':
    main()