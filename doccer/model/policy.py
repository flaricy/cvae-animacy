import torch.nn as nn
import torch
from .builder import MODELS
from .misc import get_act_module
from .distr import GaussianDistributor
import omegaconf

class ExpertModel(GaussianDistributor):
    def __init__(self, cfg : omegaconf.dictconfig.DictConfig):
        super(ExpertModel, self).__init__(cfg)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(cfg.state_dim + cfg.latent_dim, cfg.output_dim[0]))
        for i in range(1, len(cfg.output_dim)):
            self.layers.append(get_act_module(cfg.act))
            self.layers.append(nn.LayerNorm(normalized_shape=(cfg.output_dim[i - 1] + cfg.latent_dim,)))
            self.layers.append(nn.Linear(cfg.output_dim[i - 1] + cfg.latent_dim, cfg.output_dim[i]))
            
    def get_mean(self, state_t, latent_t):
        '''
        state_t: [batch_size, state_dim]
        latent_t: [batch_size, latent_dim]
        '''
        ret = self.layers[0](torch.cat([state_t, latent_t], dim=1))
        for i in range(1, len(self.layers), 3):
            ret = self.layers[i](ret)
            next_input = torch.cat([ret, latent_t], dim=1)
            next_input = self.layers[i + 1](next_input)
            ret = self.layers[i + 2](next_input)
            
        return ret
    
class GateNetwork(nn.Module):
    def __init__(self, cfg : omegaconf.dictconfig.DictConfig):
        super(GateNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(cfg.dim) - 2):
            self.layers.append(nn.Linear(cfg.dim[i], cfg.dim[i + 1]))
            self.layers.append(get_act_module(cfg.act))
        self.layers.append(nn.Linear(cfg.dim[-2], cfg.dim[-1]))
        
    def forward(self, state_t, latent_t):
        '''
        state_t: [batch_size, state_dim]
        latent_t: [batch_size, latent_dim]
        '''
        ret = torch.cat([state_t, latent_t], dim=1)
        for layer in self.layers:
            ret = layer(ret)
        '''
        ret: [batch_size, num_experts]
        '''
        return torch.softmax(ret, dim=1) # [batch_size, num_experts]

@MODELS.register_module()
class PolicyModel(nn.Module):
    def __init__(self, cfg : omegaconf.dictconfig.DictConfig):
        super(PolicyModel, self).__init__()
        cfg.gate_network.dim.append(cfg.num_experts)
        self.experts = nn.ModuleList([ExpertModel(cfg.expert_network) for _ in range(cfg.num_experts)])
        self.gate_network = GateNetwork(cfg.gate_network)
        
    def forward(self, state_t, latent_t):
        experts_ret = torch.stack([expert(state_t, latent_t) for expert in self.experts], dim=1) # [batch_size, num_experts, action_dim]
        experts_weight = self.gate_network(state_t, latent_t) # [batch_size, num_experts]
        ret = torch.sum(experts_ret * experts_weight.unsqueeze(2), dim=1) # [batch_size, action_dim]
        return ret
        
        