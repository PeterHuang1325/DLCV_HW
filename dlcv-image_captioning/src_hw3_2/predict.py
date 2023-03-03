import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv
from torch.utils.data import DataLoader

from tokenizers import Tokenizer
from PIL import Image
import argparse
import copy
from models import caption
from datasets import utils
import models
from configuration import Config
import os
import tqdm
from PIL import Image
import numpy as np
import random
from datasets.utils import nested_tensor_from_tensor_list, read_json
import json 

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--input_path', type=str, help='path to image', required=True)
parser.add_argument('--output_path', type=str, help='path to json', required=True)
parser.add_argument('--v', type=str, help='version', default='pretrained')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default='./src_hw3_2/checkpoint_1121.pth')

#setting
args = parser.parse_args()
image_path = args.input_path
out_path = args.output_path
version = args.v
checkpoint_path = args.checkpoint
MAX_DIM = 299
#configuration
config = Config()



print("Checking for checkpoint.")
if checkpoint_path is None:
    raise NotImplementedError('No model to chose from!')
else:
    if not os.path.exists(checkpoint_path):
        raise NotImplementedError('Give valid checkpoint path')
    print("Found checkpoint! Loading!")
    model,_ = caption.build_model(config)
    print("Loading Checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])


'''
About Data
'''
def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float64)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


test_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



class HW3_2_Test(Dataset):
    def __init__(self, path, transform=test_transform):
        super().__init__()
        self.files = [os.path.join(path,x) for x in os.listdir(path) if (x.endswith(".jpg")) or (x.endswith(".png"))]
        self.transform = transform
        
    def __filename__(self, idx):
        filename = self.files[idx].split('/')[-1].split('.')[0]
        return filename
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        #read images
        image = Image.open(self.files[idx])
        #transform
        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))
        
        return image.tensors.squeeze(0), image.mask.squeeze(0)
      
        
def create_caption_and_mask(start_token, max_length, pred_len):
    caption_template = torch.zeros((pred_len, max_length), dtype=torch.long)
    mask_template = torch.ones((pred_len, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


@torch.no_grad()
def evaluate(model, data_loader, device):
    #set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    #tokenizer
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    start_token = tokenizer.token_to_id("[SEP]")
    end_token = tokenizer.token_to_id("[CLS]")
    #print(dir(tokenizer))

    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True)
    #start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    #end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
    
    #model evaluation
    model.to(device)
    #model.eval()
    total = len(data_loader)
    
    with tqdm.tqdm(total=total) as pbar:
        pred_list, answer_list = [], []
        for images, masks in data_loader:
            pred_len = images.shape[0] #[32, 32, ....29] 
            caps, cap_masks = create_caption_and_mask(start_token, config.max_position_embeddings, pred_len)
            
            #images: (32, 3, 299, 299), masks: (32, 299, 299), cap: (32 ,128) 
            #samples = models.utils.NestedTensor(images, masks).to(device)
            images = images.to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)
            #outputs shape (32, 128, 30522)
            #outputs = model(images, caps, cap_masks)
            
            pbar.update(1)
            
            #predictions = copy.deepcopy(outputs)
            #config.max_position_embeddings
            
            for b in range(caps.shape[0]):
                for i in range(config.max_position_embeddings-1):
                    #text predictions
                    #predicts = predictions[:,i,:]
                    outputs = model(images[b].unsqueeze(0), caps[b].unsqueeze(0), cap_masks[b].unsqueeze(0))
                    predictions = copy.deepcopy(outputs)
                    predicts = predictions[0,i,:] #b
                    predicted_id = torch.argmax(predicts, axis=-1) #(32)
                    
                    if (predicted_id.item() == 102) or (i>70):
                        break
                    '''    
                    if predicted_id[b] == 102:
                        break
                    '''
                    caps[b, i+1] = predicted_id.item() #predicted_id[0] 
                    cap_masks[b, i+1] = False
                '''
                #check only the first sentence
                for b in range(caps.shape[0]):
                    caps[b, i+1] = predicted_id[b] 
                    cap_masks[b, i+1] = False
                '''
                #result_answer = tokenizer.decode(caps[b].tolist(), skip_special_tokens=True).capitalize()
                result_pred = tokenizer.decode(caps[b].tolist(), skip_special_tokens=True).capitalize()
                #print(result_pred.capitalize()) #capitalize the first letter

                #append to stored list
                #answer_list.append(result_answer)
                pred_list.append(result_pred)
                
        #print cider score
        print('length predictions:', len(pred_list))
        
        #cid_score = cider_score(pred_list, answer_list)
        #print('CIDEr:', cid_score)
    
    return  pred_list #, cid_score
        
def main(config):
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')
    dataset_test = HW3_2_Test(image_path, transform=test_transform)
    print(f"Test: {len(dataset_test)}")
    #data
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, config.batch_size, sampler=sampler_test, drop_last=False, num_workers=config.num_workers)
    #preds, cid_score = evaluate(model, data_loader_test, device)
    preds = evaluate(model, data_loader_test, device)
    test_filenames = [dataset_test.__filename__(f) for f in range(dataset_test.__len__())]
    #add to dictionary
    pred_dict = {}
    for p, sent in enumerate(preds):
        pred_dict[test_filenames[p]] = sent
    
    #json_name = image_path.split('/')[-1]
    with open(out_path, 'w') as outfile:
        json.dump(pred_dict, outfile)
    
if __name__ == '__main__':
    main(config)
    
