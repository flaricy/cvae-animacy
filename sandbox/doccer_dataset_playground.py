import sys 
sys.path.append(".")
from doccer.dataset.doccer import DoccerGTDataset
from doccer.utils.parse_config import load_config 
from omegaconf import OmegaConf

cfg = OmegaConf.create(load_config("example_configs/default.py"))
dataset = DoccerGTDataset(cfg.world_model_trainer.dataset)

data = dataset[0]
for key in data:
    print(f"{key}: {type(data[key])} {data[key].shape} {data[key].dtype}")