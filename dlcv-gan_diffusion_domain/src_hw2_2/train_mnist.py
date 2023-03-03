import argparse
import datetime
import torch
import wandb

from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torch.nn.functional as F
import script_utils
import matplotlib.pyplot as plt
from datasets import get_datasets, get_data_loaders
import logging
import os 
import numpy as np
from tqdm import tqdm

def main():
    args = create_argparser().parse_args()
    
    #set seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    #setting
    device = args.device
    num_workers = args.num_workers
    _dataset_dir = args.dataset_dir
    
    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))
        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        if args.log_to_wandb:
            if args.project_name is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")
            
            #wandb init
            os.environ["WANDB_START_METHOD"] = "thread"
            run = wandb.init(
                project=args.project_name,
                entity='PeterHuang1325',
                config=vars(args),
                name=args.run_name,
            )
            wandb.watch(diffusion)

        batch_size = args.batch_size
        
        
        train_dataset, val_dataset = get_datasets(_dataset_dir)
        train_loader, val_loader = get_data_loaders(train_dataset, val_dataset, batch_size, num_workers)
        
        acc_train_loss = 0

        for iteration in range(1, args.iterations + 1):
            
            progress_bar = tqdm(train_loader)
            progress_bar.set_description(f"Iteration {iteration}")
            #iteration
            #print('iteration:', iteration)
            diffusion.train()

            x, y = next(script_utils.cycle(train_loader))
            x = x.to(device)
            y = y.to(device)

            if args.use_labels:
                loss = diffusion(x, y)
            else:
                loss = diffusion(x)

            acc_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            diffusion.update_ema()
            
            if iteration % args.log_rate == 0:
                val_loss = 0
                with torch.no_grad():
                    diffusion.eval()
                    for x, y in val_loader:
                        x = x.to(device)
                        y = y.to(device)

                        if args.use_labels:
                            loss = diffusion(x, y)
                        else:
                            loss = diffusion(x)

                        val_loss += loss.item()
                
                if args.use_labels:
                    sample_list = []
                    for i in range(10):
                        torch.cuda.manual_seed_all(args.seed+i)
                        sample = diffusion.sample(10, device, y=torch.arange(10, device=device))
                        sample_list.append(sample)
                    #vstack 10x10, each class with 10 imgs
                    samples = torch.vstack(sample_list)
                else:
                    samples = diffusion.sample(100, device)
                
                #reshape tp 28x28
                samples = ((F.interpolate(samples, (28, 28)) + 1) / 2).clip(0, 1)#.permute(0, 2, 3, 1)
                #print(samples.shape) (100, 3, 28, 28)
                # Show some images during training.
                grid_img = torchvision.utils.make_grid(samples.cpu(), nrow=10)
                plt.figure(figsize=(10,10))
                plt.imshow(grid_img.permute(1, 2, 0)) #.permute(1, 2, 0)
                plt.show()
                
                val_loss /= len(val_loader)
                acc_train_loss /= args.log_rate
                #for wandb
                samples = samples.permute(0, 2, 3, 1).numpy()
                
                progress_bar.set_postfix(iteration=iteration, train_loss=acc_train_loss, val_loss=val_loss)
                
                wandb.log({
                    "val_loss": val_loss,
                    "train_loss": acc_train_loss,
                    "samples": [wandb.Image(sample) for sample in samples],
                })

                acc_train_loss = 0
            
            if iteration % args.checkpoint_rate == 0:
                
                if not os.path.exists(args.log_dir):
                    os.makedirs(args.log_dir)
                
                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-model.pth"
                optim_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"

                torch.save(diffusion.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)
        
        if args.log_to_wandb:
            run.finish()
    except KeyboardInterrupt:
        if args.log_to_wandb:
            run.finish()
        print("Keyboard interrupt, run finished early")


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    defaults = dict(
        learning_rate=2e-4,
        batch_size=128, #128
        iterations=10000, #20000
        dataset_dir = '../hw2_data/digits/mnistm/',
        num_workers=2,
        seed=1000,
        log_to_wandb=True,
        log_rate=1000, #1000
        checkpoint_rate=1000, #1000
        log_dir='./ddpm_logs/',
        project_name=None,
        run_name=run_name,

        model_checkpoint=None,
        optim_checkpoint=None,

        schedule_low=1e-4,
        schedule_high=0.02,

        device=device,
    )
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()