import sys
sys.path.append('.')

import numpy as np
import torch
import argparse
from omegaconf import OmegaConf
from doccer.engines.generator import Generator
from doccer.utils.parse_config import load_config

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--device', '-d', type=str, required=False)
    parser.add_argument('--gen_ckpt', '-p', type=str, required=True)

    args = parser.parse_args()
    return args

def set_default_dtype():
    torch.set_default_dtype(torch.float32)

if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.create(load_config(args.config))

    if args.device is not None:
        config.device = args.device

    set_default_dtype()

    generator = Generator(config, args.gen_ckpt)
    dataset_length = len(generator.dataset)
    data = generator.dataset[0]
    initial_state = data['state'][0] # (state_D)

    generated_state = generator.generate(initial_state)
    print(generated_state.shape)