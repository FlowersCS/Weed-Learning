import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=input_dim,
                               out_channels=6,
                               kernel_size=5) 
        self.maxpooling = nn.MaxPool2d(kernel_size=2,
                               stride=2) 
        self.c3 = nn.Conv2d(6,16,5) 
        self.flatten = nn.Flatten()
        self.c5 = nn.Linear(in_features=16*4*4,
                            out_features=120)
        self.f6 = nn.Linear(120,84)
        self.output = nn.Linear(84,output_dim)
    
    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.maxpooling(x)
        x = F.relu(self.c3(x))
        x = self.maxpooling(x)
        x = self.flatten(x)
        x = F.relu(self.c5(x))
        x = F.relu(self.f6(x))
        x = self.output(x)
        return x  