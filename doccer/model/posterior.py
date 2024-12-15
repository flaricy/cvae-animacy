import torch 
import torch.nn as nn 
from .distr import GaussianDistributor 
from .builder import MODELS 
from .misc import get_act_module
import omegaconf
from .prior import ConditionalPrior

@MODELS.register_module()
class ApproximatePosterior(GaussianDistributor):
    def __init__(self, cfg : omegaconf.dictconfig.DictConfig):
        super(ApproximatePosterior, self).__init__(cfg)
        self.conditional_prior = ConditionalPrior(cfg)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(cfg.state_dim, cfg.output_dim[0]))
        for i in range(1, len(cfg.output_dim)):
            self.layers.append(get_act_module(cfg.act))
            self.layers.append(nn.Linear(cfg.output_dim[i - 1] + cfg.state_dim * 2, cfg.output_dim[i]))
        
    def get_mean(self, state_t, state_t_1):
        '''
        state_t: [batch_size, state_dim]
        state_t_1: [batch_size, state_dim]
        '''
        prior_mean = self.conditional_prior.get_mean(state_t)
        ret = self.layers[0](state_t)
        for i in range(1, len(self.layers), 2):
            ret = self.layers[i](ret)
            ret = self.layers[i + 1](torch.cat([ret, state_t, state_t_1], dim=1))
            
        return ret + prior_mean