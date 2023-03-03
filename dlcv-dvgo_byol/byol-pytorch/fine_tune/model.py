import torchvision.models as models
import torch.nn as nn
import torch

#build model
def build_model(model_path, pretrained=True, fix_backbone=False, num_classes=65):
    if pretrained is True:
        print('[INFO]: Loading pre-trained weights')
        model = models.resnet50(weights=None)
        model.load_state_dict(torch.load(model_path))
        
        #remove last layer and add classifier        
        num_feat = model.fc.in_features
        features = list(model.fc.children())[:-1]
        features.extend([
            nn.Linear(num_feat, num_classes)
        ])
        
        if fix_backbone is False:
            print('[INFO]: Fine-tuning all layers...')
            
            for params in model.parameters():
                params.requires_grad = True
            
        else:
            print('[INFO]: Freezing backbone layers...')
            for params in model.named_parameters():
                #unfreeze classifier only
                if (params[0] == 'fc.weight') or (params[0] == 'fc.bias'):
                    params[1].requires_grad = True
                else:
                    params[1].requires_grad = False

    else: #not pretrained, from scratch
        print('[INFO]: Not loading pre-trained weights')
        model = models.resnet50(weights=None)
    return model
