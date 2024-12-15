import torch 
import torch.nn as nn 
from .builder import MODELS 

class MLP(nn.Module):
    def __init__(self, dims : list, act : str = 'gelu'):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 2):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if act == 'gelu':
                self.layers.append(nn.GELU())
            elif act == 'relu':
                self.layers.append(nn.ReLU())
            elif act == 'leaky_relu':
                self.layers.append(nn.LeakyReLU())
            elif act == 'elu':
                self.layers.append(nn.ELU())
                
        self.layers.append(nn.Linear(dims[-2], dims[-1]))

    def forward(self, x):
        return self.layers(x)
    
def get_act_module(act : str):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'leaky_relu':
        return nn.LeakyReLU()