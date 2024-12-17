import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
from .utils import gt_to_state

def trajectory_collection(dataloader : DataLoader, model, device, num_collection=2048):
    count = 0
    model.eval()
    for gt_data in dataloader:
        gt_states = gt_to_state(gt_data).to(device) # (B(1), T, 6)
        
        state_t = gt_states[:, 0] # (B(1), 6)
        for t in range(0, gt_states.shape[1] - 1):
            count += 1
            if count > num_collection:
                break 
            
            latent_t = model.posterior(state_t, gt_states[:, t + 1])
            action_t = model.policy(state_t, latent_t)
            state_t_1 = model.world_model(state_t, action_t)
            
            
        