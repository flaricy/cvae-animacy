import torch 
import torch.nn as nn
import omegaconf
from .builder import MODELS, build_model

@MODELS.register_module()
class ControlVAE(nn.Module):
    def __init__(self, cfg : omegaconf.dictconfig.DictConfig):
        super(ControlVAE, self).__init__()
        self.posterior = build_model(cfg.posterior)
        self.policy = build_model(cfg.policy)
        self.world_model = build_model(cfg.world_model)
        
        
    
