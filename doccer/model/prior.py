import torch 
import torch.nn as nn
from .builder import MODELS 
from .misc import get_act_module
from .distr import GaussianDistributor
import omegaconf

@MODELS.register_module()
class ConditionalPrior(GaussianDistributor):
    def __init__(self, cfg : omegaconf.dictconfig.DictConfig):
        '''
        cfg:
            state_dim: int
            output_dim: list[int]
            act: str
            std: float
        '''
        super(ConditionalPrior, self).__init__(cfg)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(cfg.state_dim, cfg.output_dim[0]))
        for i in range(1, len(cfg.output_dim)):
            self.layers.append(get_act_module(cfg.act))
            self.layers.append(nn.Linear(cfg.output_dim[i - 1] + cfg.state_dim, cfg.output_dim[i]))
            
    def get_mean(self, state_t):
        '''
        state_t: [batch_size, state_dim]
        '''
        ret = self.layers[0](state_t)
        for i in range(1, len(self.layers), 2):
            ret = self.layers[i](ret)
            ret = self.layers[i + 1](torch.cat([ret, state_t], dim=1))
            
        return ret