import torch 
import torch.nn as nn 
from .builder import MODELS 

    
def get_act_module(act : str):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'leaky_relu':
        return nn.LeakyReLU()