import sys 
sys.path.append('.')

from doccer.visualizer.state_visualizer import StateVisualizer as Visualizer
from doccer.utils.parse_config import load_config
from omegaconf import OmegaConf
import pickle
import matplotlib.pyplot as plt

config = OmegaConf.create(load_config('config/20_fps/lr_1e-5.py'))
with open("data/version3_20fps/27.pkl", "rb") as f:
    data = pickle.load(f)

visualizer = Visualizer(config.simulator)
ani = visualizer.visualize(data['state'], data['state'])
ani.save("toys/try_visualization.mp4", writer='ffmpeg', fps=config.simulator.action_fps)