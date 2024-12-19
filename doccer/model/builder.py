import omegaconf 
from ..utils.registry import Registry 

MODELS = Registry('models')

def build_model(cfg : omegaconf.dictconfig.DictConfig, use_omegaconf=True):
    return MODELS.build(cfg, use_omegaconf=use_omegaconf)