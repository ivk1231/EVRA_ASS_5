import torch
import torch.nn as nn
import torch.nn.functional as F

class CompactMNIST(nn.Module):
    def __init__(self):
        super(CompactMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 3 * 3, 10)
        
    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        x = self.bn3(F.relu(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        x = x.view(-1, 64 * 3 * 3)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1) 