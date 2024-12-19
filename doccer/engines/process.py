import torch 
import torch.nn as nn
import numpy as np
from ..dataset.doccer import DoccerGTDataset
from .simulator import Simulator
from tqdm import tqdm 
import time 
        
class TrajectoryCollector:
    def __init__(self, dataset : DoccerGTDataset, device, simulator : Simulator):
        self.dataset = dataset 
        self.dataset_length = len(self.dataset)
        self.device = device
        self.simulator = simulator
        self.simulator.initialize()
        
    def randomly_retrieve(self) -> dict:
        random_index = np.random.randint(0, self.dataset_length)
        return self.dataset[random_index]
    
    
    def np_2_tensor_w_batch_dim(self, array : np.ndarray) -> dict:
        return torch.from_numpy(array).unsqueeze(0).to(self.device)
    
    def tensor_w_batch_dim_2_np(tensor : torch.Tensor) -> np.ndarray:
        '''
        tensor : (B, T, D)
        '''
        return tensor.squeeze(0).cpu().numpy()
    
    def simulate_one_step(self, state_t, action_t):
        self.simulator.set_state(state_t)
        action_t = action_t > 0
        corrected_action = self.simulator.action_correction(action_t)
        result = self.simulator.conduct_action(corrected_action)
        state_t_1 = self.simulator.get_state()
        return state_t_1, result
    
    def collect_one_trajectory(self, clip_np : dict, model : nn.Module):
        clip_tensor_w_batch_dim = dict()
        for key in clip_np.keys():
            clip_tensor_w_batch_dim[key] = self.np_2_tensor_w_batch_dim(clip_np[key])
            
            
        state_tensor_w_batch_dim = torch.zeros((1, clip_np['state'].shape[0], clip_np['state'].shape[1])).to(self.device)
        action_tensor_w_batch_dim = torch.zeros((1, clip_np['action'].shape[0], clip_np['action'].shape[1])).to(self.device)
            
            
        state_tensor_w_batch_dim[:, 0] = clip_tensor_w_batch_dim['state'][:, 0]
        for t in range(0, clip_tensor_w_batch_dim['state'].shape[1] - 1):
            
            latent_t_tensor_w_batch_dim = model.posterior(state_tensor_w_batch_dim[:, t], clip_tensor_w_batch_dim['state'][:, t + 1])
            action_t_tensor_w_batch_dim = model.policy(state_tensor_w_batch_dim[:, t], latent_t_tensor_w_batch_dim)
            action_tensor_w_batch_dim[:, t] = action_t_tensor_w_batch_dim
            
            state_t_np = TrajectoryCollector.tensor_w_batch_dim_2_np(state_tensor_w_batch_dim[:, t])
            action_t_np = TrajectoryCollector.tensor_w_batch_dim_2_np(action_tensor_w_batch_dim[:, t])
            
            state_t_1_np, result = self.simulate_one_step(state_t_np, action_t_np)
            
            if result is not None:
                break
            
            state_tensor_w_batch_dim[:, t + 1] = self.np_2_tensor_w_batch_dim(state_t_1_np)
            
        state_np = TrajectoryCollector.tensor_w_batch_dim_2_np(state_tensor_w_batch_dim)
        action_np = TrajectoryCollector.tensor_w_batch_dim_2_np(action_tensor_w_batch_dim)
    
        if t != clip_tensor_w_batch_dim['state'].shape[1] - 1:
            state_np = state_np[:t+1]
            action_np = action_np[:t+1]
            clip_np['state'] = clip_np['state'][:t+1]
            
        return dict(
            state=state_np,
            action=action_np,
            gt_state=clip_np['state']
        )
    
    def collect_trajectory(self, model : nn.Module, num_trajectories : int):
        ret_pieces = []
        for _ in tqdm(range(num_trajectories), desc='Collecting trajectories'):
            clip_np = self.randomly_retrieve()
            ret_pieces.append(self.collect_one_trajectory(clip_np, model))
            
        return ret_pieces