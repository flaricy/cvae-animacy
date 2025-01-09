import sys 
sys.path.append(".")
from doccer.dataset.doccer import DoccerDataset 
from doccer.utils.parse_config import load_config 
from omegaconf import OmegaConf

cfg = OmegaConf.create(load_config("config/20_fps/lr_1e-5.py"))
dataset = DoccerDataset(cfg.dataset)

print(len(dataset))