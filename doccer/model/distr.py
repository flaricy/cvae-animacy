from .misc import MLP 
from .builder import MODELS 
import omegaconf 
import torch 
import torch.nn as nn 

class GaussianDistributor(nn.Module):
    def __init__(self, cfg : omegaconf.dictconfig.DictConfig):
        super(GaussianDistributor, self).__init__()
        self.std = cfg.std
        
    def get_mean(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
    
    def forward(self, *args, **kwargs):
        mean = self.get_mean(*args, **kwargs)
        sample_ret = torch.rand_like(mean)
        ret = sample_ret * self.std + mean 
        return ret
        