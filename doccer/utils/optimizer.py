import torch
import torch.optim as optim 
import torch.nn as nn
from .registry import Registry
import omegaconf 
from omegaconf import OmegaConf

OPTIMIZERS = Registry('optimizer')

OPTIMIZERS.register_module(module=optim.Adam, name='Adam')
OPTIMIZERS.register_module(module=optim.SGD, name='SGD')
OPTIMIZERS.register_module(module=optim.AdamW, name='AdamW')


def build_optimizer(cfg : omegaconf.dictconfig.DictConfig, params : list) -> optim.Optimizer:
    dict_cfg = OmegaConf.to_container(cfg)
    dict_cfg['params'] = params
    return OPTIMIZERS.build(cfg=dict_cfg, use_omegaconf=False)