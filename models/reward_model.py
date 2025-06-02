import torch
from torch import nn

class RewardFCModel(nn.Module):
    def __init__(self,input_dim=5,hidden_dim=64,num_hidden_layers=2):
        super().__init__()
        # Define your layers here
        self.activation=torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        #network for reward prediction
        self.fc_0=torch.nn.Linear(input_dim, hidden_dim)
        self.fch = torch.nn.Sequential(*[
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),  # Adjust input size to N_HIDDEN
                self.activation
            ) for _ in range(num_hidden_layers - 1)
        ])
        self.fc_n=torch.nn.Linear(hidden_dim, 1)


    def forward(self,x):
        r=self.fc_0(x)
        r=self.activation(r)
        r=self.fch(r)
        r=self.fc_n(r)
        r=self.tanh(r)
        return r