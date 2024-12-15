import torch 
import torch.nn as nn 
from .builder import MODELS 
from .misc import get_act_module
from .distr import GaussianDistributor
import omegaconf

@MODELS.register_module()
class WorldModel(nn.Module):
    def __init__(self, cfg : omegaconf.dictconfig.DictConfig):
        super(WorldModel, self).__init__()
        self.layers = nn.Sequential()
        for i in range(len(cfg.dim) - 2):
            self.layers.append(nn.Linear(cfg.dim[i], cfg.dim[i + 1]))
            self.layers.append(get_act_module(cfg.act))
        self.layers.append(nn.Linear(cfg.dim[-2], cfg.dim[-1]))
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, state_t, action_t):
        '''
        state_t: [batch_size, state_dim]
        action_t: [batch_size, action_dim]
        '''
        x = torch.cat([state_t, action_t], dim=1) # [batch_size, state_dim + action_dim]
        return self.layers(x)