import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torch.optim as optim
from torchvision.transforms import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
from prune_layers import MaskedConv2dDynamic
from prune_layers import MaskedLinearDynamic
#VGG19 with pruning layers. Modified from VGG19

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG_prune(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features
        self.fc1 = MaskedLinearDynamic(512, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = MaskedLinearDynamic(4096, 4096)
        self.fc3 = nn.Linear(4096, num_class)
        

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
    
        return output
    
    def update_masks(self, c_rate):
        #self.conv1.update_mask()
        #self.conv2.update_mask()
        for i in range(len(self.features)):
            if isinstance(self.features[i], MaskedConv2dDynamic):
                self.features[i].make_weight_copy()
                self.features[i].update_mask(c_rate)
        self.fc1.update_mask(c_rate)
        self.fc2.update_mask(c_rate)
        #self.fc3.update_mask()
        
    def set_weight_back(self):
        print('=======reset weight========')
        for i in range(len(self.features)):
            if isinstance(self.features[i], MaskedConv2dDynamic):
                self.features[i].set_weight_back()
               
    
        
    def set_flags(self, flag):
        #self.conv1.set_maskflag(flag)
        #self.conv2.set_maskflag(flag)
        for i in range(len(self.features)):
            if isinstance(self.features[i], MaskedConv2dDynamic):
                self.features[i].set_maskflag(flag)
        self.fc1.set_maskflag(flag)
        self.fc2.set_maskflag(flag)
        #self.fc3.set_maskflag(flag)



def make_pruned_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [MaskedConv2dDynamic(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        
        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    
    return nn.Sequential(*layers)

def vgg11_bn_prune():
    return VGG_prune(make_pruned_layers(cfg['A'], batch_norm=True))

def vgg13_bn_prune():
    return VGG_prune(make_pruned_layers(cfg['B'], batch_norm=True))

def vgg16_bn_prune():
    return VGG_prune(make_pruned_layers(cfg['D'], batch_norm=True))

def vgg19_bn_prune():
    return VGG_prune(make_pruned_layers(cfg['E'], batch_norm=True))