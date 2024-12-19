import sys 
sys.path.append('.')

import numpy as np 
import torch 
import argparse
from doccer.engines.trainer import Trainer 
from doccer.utils.parse_config import load_config 
from omegaconf import OmegaConf

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--device', '-d', type=str, required=False)
    
    args = parser.parse_args()
    return args 

def set_default_dtype():
    torch.set_default_dtype(torch.float32)
    
def set_debug_mode():
    torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.create(load_config(args.config))
    
    if args.device is not None:
        config.device = args.device 
        
    set_default_dtype()
    set_debug_mode()
        
    trainer = Trainer(config)
    trainer.train()