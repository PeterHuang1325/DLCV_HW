# Import necessary packages.
import torch
import numpy as np
import argparse
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import imageio
import time
import os
from tqdm.auto import tqdm
from model import build_model
import pandas as pd
import scipy.ndimage
from matplotlib import colors as mcolors



def read_options():
    # construct the argument parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--input_path', type=str,
        help='input data directory'
    )
    parser.add_argument(
        '--output_path', type=str,
        help='output data directory'
    )
    parser.add_argument(
        '--epochs', type=int, default=60,
        help='Number of epochs to train our network for'
    )
    parser.add_argument(
        '--pretrained', action='store_false', default=True,
        help='Whether to use pretrained weights or not'
    )

    parser.add_argument(
        '--image_size', type=int, default=512,
        help='image size for training the model'
    )
    parser.add_argument(
         '--batch_size', type=int, default=12,
        help='batch size for training'
    )
    parser.add_argument(
        '--num_classes', type=int, default=7,
        help='number of label classes'
    )
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='parellel num workers'
    )
    parser.add_argument(
        '--seed', type=int, default=1000,
        help='seed for experiment'
    )

    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))
    
    return parsed 

class HW1Dataset(Dataset):
    def __init__(self, filepath):
        #self.mask_file = sorted([os.path.join(filepath, file) for file in os.listdir(filepath) if file.endswith('.png')])
        self.imgs_file = sorted([os.path.join(filepath, file) for file in os.listdir(filepath) if file.endswith('.jpg')])
        
    def __len__(self):
        return len(self.imgs_file)
  
    def __filename__(self, idx):
        return self.imgs_file[idx].split('/')[-1].split('_')[0]
    
    def __getitem__(self, idx):
        '''
        masks = torch.empty((self.__len__(), 512, 512))
        #masks
        mask = imageio.v2.imread(self.mask_file[idx])
        mask = (mask >= 128).astype(int)
        
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        
        masks[idx, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[idx, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[idx, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[idx, mask == 2] = 3  # (Green: 010) Forest land 
        masks[idx, mask == 1] = 4  # (Blue: 001) Water 
        masks[idx, mask == 7] = 5  # (White: 111) Barren land 
        masks[idx, mask == 0] = 6  # (Black: 000) Unknown
        
        mask = masks[idx]
        #mask = mask.astype(np.uint8) #for transform to PIL
        '''
        #images
        image = torch.from_numpy(imageio.v2.imread(self.imgs_file[idx])/255).permute(2,0,1).type(torch.float32)
        #data augmentation
        #image, mask = self.transform(image=image, mask=mask)
        image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
        #mask = mask*255
        
        return image#, mask

# data visualization masks
voc_cls = {'urban':0, 
           'rangeland': 2,
           'forest':3,  
           'unknown':6,  
           'barreb land':5,  
           'Agriculture land':1,  
           'water':4} 
cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0],
}

    

#denormalize
def denormalize(images):
    means = torch.tensor([0.485, 0.456, 0.406])
    stds = torch.tensor([0.229, 0.224, 0.225])
    return images * stds + means


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
    eval_set = HW1Dataset(args['input_path'])
    print(f"[INFO]: Number of validation images: {len(eval_set)}")

    # Load the training and validation data loaders.
    eval_loader = DataLoader(eval_set, batch_size=args['batch_size'],shuffle=False, num_workers=args['num_workers'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Computation device: {device}")
    
    #build model
    model = build_model(model_name='DeeplabV3', num_classes=args['num_classes']).to(device)
    model.load_state_dict(torch.load('./src_hw1_2/model_problem2_DeeplabV3_730.pth')['model_state_dict'])
    
    model.eval()
    val_out_cut = []
    
    with torch.no_grad():
        for step, image in tqdm(enumerate(eval_loader), total=len(eval_loader)):
            image = image.to(device)
            #mask = mask.to(device)
            
            outputs = model(image)
                
            #out_cut = np.copy(outputs.data.cpu().numpy())    
            out_cut = torch.argmax(outputs['out'], dim=1)
            val_out_cut.append(out_cut.data.cpu().numpy())
            
        
    #stack full predicted masks
    val_outs = np.vstack(val_out_cut) #(257, 512, 512)

    print('EVALUATION COMPLETE')
    print('-'*50)
    save_path = args['output_path']
    
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    '''
    os.makedirs(save_path, exist_ok=True)
    
    for idx, data in enumerate(eval_set):
        img = 255*denormalize(data.permute(1,2,0))
        masks = val_outs[idx] #(512, 512)
        #print(img.shape)
        #print(masks.shape)
    
        image_name = eval_set.__filename__(idx)
        
        cs = np.unique(masks)
        mask = np.zeros((img.shape[0], img.shape[1], 3))
        for c in cs:
            ind = np.where(masks==c)
            mask[ind[0], ind[1]] = cmap[c]

        imageio.imsave(args['output_path']+f'/{image_name}.png', np.uint8(mask))

if __name__ == '__main__':
    cmap = cls_color
    main()

