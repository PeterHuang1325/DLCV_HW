import torch
from torch.utils.data import DataLoader
import math
import numpy as np
import time
import sys
import os
import tqdm
from models import utils, caption
from datasets import dataset
from configuration import Config
from engine import train_one_epoch, evaluate


def main(config):
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')
    #load_path = config.load_path
    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    _, criterion = caption.build_model(config)
    #load pretrained model
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
    #model.load_state_dict(torch.load(os.path.join(load_path, 'weight493084032.pth')))
    model_dict = model.state_dict()
    #let decoder train from scratch initialization
    '''
    decode_lyrs = [k for k in model_dict.keys() if 'decoder' in k]
    for lyr in decode_lyrs:
        #model_dict[lyr] = model_dict[lyr] + torch.randn(model_dict[lyr].shape)*0.01 #=
        model_dict[lyr] = torch.randn(model_dict[lyr].shape)*0.01
    '''
    #set encoder and decoder both random initial
    lyrs = [k for k in model_dict.keys() if ('encoder' in k) or ('decoder' in k)]
    for lyr in lyrs:
        #model_dict[lyr] = model_dict[lyr] + torch.randn(model_dict[lyr].shape)*0.01 #=
        model_dict[lyr] = torch.randn(model_dict[lyr].shape)*0.01
        
    model.load_state_dict(model_dict)
    #freeze resnet backbone and encoder
    model.backbone.requires_grad_(False)
    #model.transformer.encoder.requires_grad_(False)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)

    dataset_train = dataset.build_dataset(config, mode='training')
    dataset_val = dataset.build_dataset(config, mode='validation')
    print(f"Train: {len(dataset_train)}")
    print(f"Valid: {len(dataset_val)}")

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config.batch_size, drop_last=True
    )
    #data loader
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers)
    data_loader_val = DataLoader(dataset_val, config.batch_size, sampler=sampler_val, drop_last=False, num_workers=config.num_workers)

    if os.path.exists(config.checkpoint):
        print("Loading Checkpoint...")
        checkpoint = torch.load(config.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.start_epoch = checkpoint['epoch'] + 1

    print("Start Training..")
    best_loss = math.inf
    for epoch in range(config.start_epoch, config.epochs):
        print(f"Epoch: {epoch}")
        epoch_loss = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, config.clip_max_norm)
        lr_scheduler.step()
        print(f"Training Loss: {epoch_loss}")

        validation_loss = evaluate(model, criterion, data_loader_val, device)
        print(f"Validation Loss: {validation_loss}")
        
        #save best model
        if validation_loss < best_loss:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, config.checkpoint) #config.checkpoint, './ckpt_{epoch}.pth'
        best_loss = validation_loss


if __name__ == "__main__":
    config = Config()
    main(config)
