from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv

from PIL import Image
import numpy as np
import random
import os
from tokenizers import Tokenizer


from .utils import nested_tensor_from_tensor_list, read_json

MAX_DIM = 299


def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
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



class HW3_2_Caption(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='training'):
        super().__init__()

        self.root = root
        self.transform = transform
        
        self.annot = [(val['image_id'], val['caption']) for val in ann['annotations']]
        
        #file dict: id2filename
        self.file_dict = {}
        for file in ann['images']:
            self.file_dict[file['id']] = file['file_name'] 
        #self.filename = [file['file_name'] for file in ann['images']]
        
        if mode == 'validation':
            self.annot = self.annot
        if mode == 'training':
            self.annot = self.annot[: limit]

        self.tokenizer  = Tokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length + 1
        
    def __filename__(self, idx):
        image_id, caption = self.annot[idx]
        filename = self.file_dict[image_id]
        return filename
    
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
        
        caption_encoded = self.tokenizer.encode(caption)
        #caption
        caption_og = np.array(caption_encoded.ids)
        diff = self.max_length-len(caption_og)
        caption = np.pad(caption_og, (0,diff), 'constant', constant_values=0)
        
        #cap mask
        attention_mask = np.array(caption_encoded.attention_mask)
        attention_mask = np.pad(attention_mask, (0,diff), 'constant', constant_values=0)
        cap_mask = (1 - attention_mask).astype(bool)
        
        '''
        caption_encoded = self.tokenizer.encode_plus(
            caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True)

        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (
            1 - np.array(caption_encoded['attention_mask'])).astype(bool)
        '''
        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask



def build_dataset(config, mode='training'):
    if mode == 'training':
        train_dir = os.path.join(config.dir, 'images/train')
        train_file = os.path.join(
            config.dir, 'train.json')
        data = HW3_2_Caption(train_dir, read_json(
            train_file), max_length=config.max_position_embeddings, limit=config.limit, transform=train_transform, mode='training')
        return data

    elif mode == 'validation':
        val_dir = os.path.join(config.dir, 'images/val')
        val_file = os.path.join(
            config.dir,'val.json')
        data = HW3_2_Caption(val_dir, read_json(
            val_file), max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform, mode='validation')
        return data
    
    else:
        raise NotImplementedError(f"{mode} not supported")