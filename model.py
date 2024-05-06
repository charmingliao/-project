import socket
import dill
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms
import multiprocessing
from torch.utils.data import Dataset, DataLoader
'''class Net(nn.Module):
    def __init__(self, num_classes = 2, pretrined = False):
         super(Net, self).__init__()
         self.resnet = models.resnet50(pretrained=True)
         self.num_ftrs = self.resnet.fc.in_features
         self.resnet.fc = nn.Sequential(
            nn.Linear(self.num_ftrs,256),
            nn.ReLU(),
            nn.Dropout(0.3),    
            nn.Linear(256,10)
         )
    def forward(self,x):
        x = self.resnet(x)
        return x'''
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 8 * 8, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x