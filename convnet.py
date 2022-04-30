import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes=16, drp0=0.1, drp1=0.1):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(4608, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.drp0 = nn.Dropout(p=drp0)        
        self.drp1 = nn.Dropout(p=drp1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drp0(out)        
        out = self.fc1(out)
        out = F.leaky_relu(out,0.2) 
        out = self.drp1(out)
        out = self.fc2(out)
        return out
