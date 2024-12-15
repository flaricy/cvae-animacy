import torch 
import torch.nn as nn
import omegaconf
from .builder import MODELS, build_model

@MODELS.register_module()
class ControlVAE(object):
    def __init__(self, cfg : omegaconf.dictconfig.DictConfig):
        super(ControlVAE, self).__init__()
        self.models = dict(
            posterior=build_model(cfg.posterior),
            policy=build_model(cfg.policy),
            world_model=build_model(cfg.world_model),
        )
        
    def to(self, device):
        for key in self.models:
            self.models[key].to(device)
            
    def train(self):
        for key in self.models:
            self.models[key].train()
            
    def eval(self):
        for key in self.models:
            self.models[key].eval()
