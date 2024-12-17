import torch.optim.lr_scheduler as lr_scheduler 
import torch.optim as optim 
from .registry import Registry
import omegaconf
from omegaconf import OmegaConf 

SCHEDULERS = Registry('scheduler')

SCHEDULERS.register_module(module=lr_scheduler.StepLR, name='StepLR')
SCHEDULERS.register_module(module=lr_scheduler.MultiStepLR, name='MultiStepLR')
SCHEDULERS.register_module(module=lr_scheduler.ExponentialLR, name='ExponentialLR')
SCHEDULERS.register_module(module=lr_scheduler.CosineAnnealingLR, name='CosineAnnealingLR')

def build_scheduler(cfg : omegaconf.dictconfig.DictConfig, optimizer : optim.Optimizer) -> lr_scheduler.LRScheduler:
    dict_cfg = OmegaConf.to_container(cfg)
    dict_cfg['optimizer'] = optimizer
    return SCHEDULERS.build(cfg=dict_cfg, use_omegaconf=False)