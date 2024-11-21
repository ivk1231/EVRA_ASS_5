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
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1) 