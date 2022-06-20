import torch
from torch import nn
import torch.nn.functional as f

class SampleNet(torch.nn.Module):
    def __init__(self):
        super(SampleNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 1000)
        self.fc2 = torch.nn.Linear(1000, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)

        return f.log_softmax(x, dim=1)