from ..utils.registry import Registry
import omegaconf

DATASETS = Registry('dataset')

def build_dataset(cfg : omegaconf.dictconfig.DictConfig):
    return DATASETS.build(cfg)