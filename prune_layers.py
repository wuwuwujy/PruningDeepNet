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

#Masked linear layer
        
class MaskedLinearDynamic(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinearDynamic, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
        self.mask = Variable(self.weight.data,requires_grad=False).cuda()
    
    def set_mask(self, mask):
        self.mask = mask
        self.mask_flag = True
    
    def set_maskflag(self, mask_flag):
        self.mask_flag = mask_flag
    
    def get_mask(self):
        return self.mask
    
    def update_mask(self, c_rate):
        mean = torch.mean(torch.abs(self.weight.data)).item()
        std = torch.std(torch.abs(self.weight.data)).item()
        ak = (mean - c_rate*std)
        bk = (mean + c_rate*std)
        self.mask[abs(self.weight.data)<ak] = 0
        self.mask[abs(self.weight.data)>bk] = 1
        #remove parameters less significant according to std
        #removing rate controled by c_rate
        self.weight.data = self.weight.data*self.mask.data

    
    def forward(self, x):
        if self.mask_flag == True:
            return F.linear(x, self.weight*self.mask, bias=None)
        else:
            return F.linear(x, self.weight, bias=None)


#Masked convolution layer

class MaskedConv2dDynamic(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, bias=True):
        super(MaskedConv2dDynamic, self).__init__(in_channels, out_channels, 
            kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False
        self.mask = Variable(self.weight.data,requires_grad=False).cuda()
    
    def set_mask(self, mask):
        self.mask = mask
        self.mask_flag = True
    def set_maskflag(self, mask_flag):
        self.mask_flag = mask_flag
    
    def get_mask(self):
        return self.mask
    
    def update_mask(self, c_rate):
        mean = torch.mean(torch.abs(self.weight.data)).item()
        std = torch.std(torch.abs(self.weight.data)).item()
        ak = (mean - c_rate*std)
        bk = (mean + c_rate*std)
        self.mask[abs(self.weight.data)<ak] = 0
        self.mask[abs(self.weight.data)>bk] = 1
        #remove parameters less significant according to std
        #removing rate controled by c_rate
        #tried remove by channel but give worse result
        self.weight.data = self.weight.data*self.mask.data
    def forward(self, x):
        if self.mask_flag == True:
            return F.conv2d(x, self.weight*self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

