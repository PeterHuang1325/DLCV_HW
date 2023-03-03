import torchvision.models as models
import torch.nn as nn


#build model
def build_model(pretrained=True, fine_tune=True, num_classes=50):
    if pretrained is True:
        print('[INFO]: Loading pre-trained weights')
        model = models.efficientnet_v2_s(pretrained=pretrained, weights='IMAGENET1K_V1')
    
        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print('[INFO]: Freezing hidden layers...')
            for params in model.parameters():
                params.requires_grad = False
        
    else: #not pretrained, from scratch
        print('[INFO]: Not loading pre-trained weights')
        #model = models.resnet50(weights=None)
        model = models.resnet50()
    return model