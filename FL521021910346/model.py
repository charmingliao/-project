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
    

def train(model, dataloader, num_epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for e in range(num_epochs):
        for features, labels in dataloader:
            optimizer.zero_grad()
            probs = model(features)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()

def test(global_model, test_loader):
    all_predictions = []
    all_labels = []

    global_model.eval()

    with torch.no_grad():
        for features, labels in test_loader:
            preds = global_model(features)
            predicted_classes = torch.argmax(preds, dim=-1)
            predicted_classes = predicted_classes.tolist()
            labels = labels.tolist()

            all_predictions.extend(predicted_classes)
            all_labels.extend(labels)

    all_predictions = torch.tensor(all_predictions)
    all_labels = torch.tensor(all_labels)

    accuracy = (all_predictions == all_labels).float().mean().item()
    
    return accuracy
