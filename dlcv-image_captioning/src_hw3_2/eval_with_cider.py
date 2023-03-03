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
from cider import cider_score

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--input_path', type=str, help='path to image', required=True)
parser.add_argument('--output_path', type=str, help='path to json', required=True)
parser.add_argument('--v', type=str, help='version', default='pretrained')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default='./checkpoint.pth')

#setting
args = parser.parse_args()
image_path = args.input_path
out_path = args.output_path
version = args.v
checkpoint_path = args.checkpoint
MAX_DIM = 299
#configuration
config = Config()

if version == 'v1':
    model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
elif version == 'v2':
    model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
elif version == 'v3':
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
else:
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

train_transform = tv.transforms.Compose([
    RandomRotation(),
    tv.transforms.Lambda(under_max),
    tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                              0.8, 1.5], saturation=[0.2, 1.5]),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



class HW3_2_eval(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=val_transform):
        super().__init__()

        self.root = root
        self.transform = transform
        
        self.annot = [(val['image_id'], val['caption']) for val in ann['annotations']]
        
        #file dict: id2filename
        self.file_dict = {}
        for file in ann['images']:
            self.file_dict[file['id']] = file['file_name'] 
        #self.filename = [file['file_name'] for file in ann['images']]
        

        self.tokenizer  = Tokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length + 1
        
    def __filename__(self, idx):
        image_id, caption = self.annot[idx]
        file_name = self.file_dict[image_id].split('.')[0]
        return file_name
    
    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):        
        image_id, caption = self.annot[idx]
        filename = self.file_dict[image_id]
        #print(image_id, caption)
        image = Image.open(os.path.join(self.root, filename))
        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))
        #print(dir(self.tokenizer.encode(caption)))
        caption_encoded = self.tokenizer.encode(caption)
        #caption
        caption_og = np.array(caption_encoded.ids)
        diff = self.max_length-len(caption_og)
        caption = np.pad(caption_og, (0,diff), 'constant', constant_values=0)
        
        #cap mask
        attention_mask = np.array(caption_encoded.attention_mask)
        attention_mask = np.pad(attention_mask, (0,diff), 'constant', constant_values=0)
        cap_mask = (1 - attention_mask).astype(bool)
        #1-attention mask
        #cap_mask = (1 - np.array(caption_encoded.attention_mask)).astype(bool)
        #print(caption.shape, cap_mask.shape)
        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask
    
    
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
        pred_list, answer_list = [], [] #anser_list: [[],[],[],...]
        ans_temp = [] #[..., ..., ...]
        cnt = 1
        for images, masks, caps_true, cap_masks_true in data_loader:
            #images: (1, 3, 299, 299), masks: (1, 299, 299), cap: (1 ,129) 
            caps_true = caps_true.to(device)
            cap_masks_true = cap_masks_true.to(device)
            
            pred_len = images.shape[0] #[32, 32, ....29] 
            caps, cap_masks = create_caption_and_mask(start_token, config.max_position_embeddings, pred_len)
            
        
            images = images.to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)
            pbar.update(1)
            
            
            #answer append 
            result_answer = tokenizer.decode(caps_true[0].tolist(), skip_special_tokens=True).capitalize()
            ans_temp.append(result_answer)
            #print(f'cnt: {cnt}', ans_temp)
            if cnt % 5 == 0:
                for i in range(config.max_position_embeddings-1):
                    #text predictions
                    #predicts = predictions[:,i,:]
                    outputs = model(images[0].unsqueeze(0), caps[0].unsqueeze(0), cap_masks[0].unsqueeze(0))
                    
                    cross_weights = model.transformer.decoder.layers[-1]
                    
                    predictions = copy.deepcopy(outputs)
                    predicts = predictions[0,i,:] #b
                    predicted_id = torch.argmax(predicts, axis=-1) #(1)
                    
                    if predicted_id.item() == 102:
                        break

                    caps[0, i+1] = predicted_id.item() #predicted_id[0] 
                    cap_masks[0, i+1] = False
                    
                #decode a sentence
                result_pred = tokenizer.decode(caps[0].tolist(), skip_special_tokens=True).capitalize()
                
                pred_list.append(result_pred)
                answer_list.append(ans_temp) #[..., ..., ...]
                ans_temp = []  #reseet the list to []
            cnt += 1
            
        #print cider score
        print('length predictions:', len(pred_list))
        
        cid_score = cider_score(pred_list, answer_list)
        print('CIDEr:', cid_score)
    
    return  pred_list, cid_score

def main(config):
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')
    
    eval_dir = os.path.join(config.dir, 'images/val')
    eval_file = os.path.join(config.dir,'val.json')
    dataset_test = HW3_2_eval(eval_dir, read_json(
            eval_file), max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform)
    
    test_filenames = [dataset_test.__filename__(f) for f in range(0, len(dataset_test), 5)]
    test_filenames = list(dict.fromkeys(test_filenames)) #remove duplicate
    
    print(f"Test: {len(dataset_test)}")
        
    #data
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, batch_size=1, sampler=sampler_test, drop_last=False, num_workers=config.num_workers)
    preds, cid_score = evaluate(model, data_loader_test, device)
    test_filenames = [dataset_test.__filename__(f) for f in range(0, len(dataset_test), 5)]

    #add to dictionary
    pred_dict = {}
    for p, sent in enumerate(preds):
        pred_dict[test_filenames[p]] = sent
    
    #json_name = image_path.split('/')[-1]
    with open(out_path, 'w') as outfile:
        json.dump(pred_dict, outfile)

if __name__ == '__main__':
    main(config)
