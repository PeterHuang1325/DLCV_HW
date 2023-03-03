import torchvision.models as models
import torch.nn as nn

#model from scratch
class Scratch_Model(nn.Module):
    def __init__(self, num_classes=50):
        super(Scratch_Model, self).__init__()
        self.conv1 =  nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)
        
        self.conv3 =  nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048,1024)
        self.relu4 = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes) 
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu4(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out

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
        model = Scratch_Model(num_classes=50)
    return model