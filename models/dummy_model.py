import torch
from torch import nn

class DummyModel(nn.Module):
    def __init__(self,input_dim=5):
        super().__init__()
        # Define your layers here
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Define the forward pass here
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x