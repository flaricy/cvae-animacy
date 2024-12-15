from .builder import MODELS
import torch 
import torch.nn as nn
import omegaconf
from .distr import GaussianDistributor

@MODELS.register_module()
class ControlVAE(nn.Module):
    def __init__(self, cfg : omegaconf.dictconfig.DictConfig):
        super(ControlVAE, self).__init__()
        self.conditional_prior = GaussianDistributor(cfg.conditional_prior)
        self.policy = GaussianDistributor(cfg.)