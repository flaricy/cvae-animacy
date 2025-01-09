import sys 
sys.path.append('doccer')

from dataset.doccer import DoccerDataset
from omegaconf import OmegaConf

config = OmegaConf.load("config/toy_config.yaml")

dataset = DoccerDataset(config.dataset)

ret = dataset[0]
for key in ret.keys():
    print(key, ret[key].shape)
