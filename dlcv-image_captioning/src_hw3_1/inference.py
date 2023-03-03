import torch
import clip
from PIL import Image
import json
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import pandas as pd
import os 
import imageio
import json
import argparse


def read_options():
    # construct the argument parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--data_path', type=str,
        help='input data directory'
    )
    parser.add_argument(
         '--json_path', type=str,
        help='json file directory'
    )
    parser.add_argument(
         '--output_path', type=str,
        help='output file directory'
    )

    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))
    
    return parsed 
#Dataset
class HW3_1_Dataset(Dataset):
    def __init__(self, path, tfm = None ,files = None):
        super(HW3_1_Dataset).__init__()
        self.path = path
        self.files = [os.path.join(path,x) for x in os.listdir(path) if x.endswith(".png")]
        #self.files = random.sample(self.files, len(self.files))
        if files != None:
            self.files = files
        #print(f"One {path} sample",self.files[0]) 
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
    
    def __filename__(self,idx):
        return self.files[idx].split('/')[-1]
    
    def __getitem__(self,idx):
        fname = self.files[idx]
        #read image
        im = np.array(imageio.v2.imread(fname))
        im = Image.fromarray(im)
        #im = torch.from_numpy(im).permute(2,0,1)
        #transform
        if self.transform != None:
            im = self.transform(im)
        return im


def main():
    #read parsed
    args = read_options()
    
    #load json
    with open(args['json_path'], newline='') as jsonfile:
        id2label = json.load(jsonfile)
    
    #load data
    eval_set = HW3_1_Dataset(os.path.join(args['data_path']), tfm=None)
    
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    #set to eval mode
    model.eval()
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in list(id2label.values())]).to(device)
    correct = 0
    prediction = []
    
    #evaluation
    for i, image in tqdm(enumerate(eval_set), total=len(eval_set)):
        
        image_input = preprocess(image).unsqueeze(0).to(device)
        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # Pick the top 1 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)
        prediction.append(indices.cpu().numpy()[0])

    print('-'*5+'Zero-shot complete'+'-'*5)
    
    #output to csv
    #get files
    eval_files = [eval_set.__filename__(idx) for idx in range(len(eval_set))]
    out_df = pd.DataFrame(np.array([eval_files, prediction], dtype=object).T, columns=['filename', 'label'])
    out_df.to_csv(os.path.join(args['output_path']), index=False)

if __name__ == '__main__':
    main()