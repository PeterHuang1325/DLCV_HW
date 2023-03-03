import argparse
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import script_utils
import os

def main():
    
    
    args = create_argparser().parse_args()
    device = args.device
    
    #set seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        diffusion.load_state_dict(torch.load('./DDPM_best.pth')) 
        
        #create save dir if not exists
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        
        if args.use_labels:
            for label in range(10):
                y = torch.ones(args.num_images // 10, dtype=torch.long, device=device) * label
                samples = diffusion.sample(args.num_images // 10, device, y=y)

                for image_id in range(len(samples)):
                    image = ((F.interpolate(samples[image_id].unsqueeze(0), (28, 28)) + 1) / 2).clip(0, 1).squeeze()
                    torchvision.utils.save_image(image, f"{args.output_path}/{label}_{str(image_id+1).zfill(3)}.png")
        else:
            samples = diffusion.sample(args.num_images, device)

            for image_id in range(len(samples)):
                image = ((F.interpolate(samples[image_id].unsqueeze(0), (28, 28)) + 1) / 2).clip(0, 1).squeeze()
                torchvision.utils.save_image(image, f"{args.output_path}/{image_id}.png")
    except KeyboardInterrupt:
        print("Keyboard interrupt, generation finished early")


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(num_images=1000, device=device)
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--schedule_low", type=float, default=1e-4) 
    parser.add_argument("--schedule_high", type=float, default=0.02)
    #parser.add_argument("--use_labels", type=action='store_false', default=True)
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()