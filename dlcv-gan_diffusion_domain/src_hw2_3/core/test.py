import torch.utils.data
import torch.nn as nn
import numpy as np

#register hook
'''
features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook
'''

def test(model, data_loader, device, flag):
    """Evaluate model for dataset."""
    # set eval state for Dropout and BN layers
    model.eval()

    # init loss and accuracy
    loss_ = 0.0
    acc_ = 0.0
    acc_domain_ = 0.0
    n_total = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()
    #feat_full = []
    # evaluate network
    for (images, labels) in data_loader:
        images = images.to(device)
        labels = labels.to(device)  #labels = labels.squeeze(1)
        size = len(labels)
        if flag == 'target':
            labels_domain = torch.ones(size).long().to(device)
        else:
            labels_domain = torch.zeros(size).long().to(device)
        
        #model.feature[8].register_forward_hook(get_features('feature'))
        preds, domain = model(images, alpha=0)
        #feat_full.append(features['feature'].cpu().numpy())
        loss_ += criterion(preds, labels).item()
        
        pred_cls = preds.data.max(1)[1]
        pred_domain = domain.data.max(1)[1]
        acc_ += pred_cls.eq(labels.data).sum().item()
        acc_domain_ += pred_domain.eq(labels_domain.data).sum().item()
        n_total += size

    loss = loss_ / n_total
    acc = acc_ / n_total
    acc_domain = acc_domain_ / n_total
    
    print("Domain={}, Avg Loss = {:.6f}, Avg Accuracy = {:.2%}, {}/{}, Avg Domain Accuracy = {:2%}".format(flag, loss, acc, acc_, n_total, acc_domain))

    return loss, acc, acc_domain#, feat_full 
