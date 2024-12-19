import torch
import torch.nn as nn 
import omegaconf 

class KLDivergLoss(nn.Module):
    def __init__(self, cfg:omegaconf.dictconfig.DictConfig):
        super(KLDivergLoss, self).__init__()
        self.decay_factor = cfg.decay_factor_gamma
    def forward(self, generate_delta_latent):
        '''
        generate_delta_latent: (B, T-1, latent_dim)
        '''
        norm = torch.sum(generate_delta_latent ** 2, dim=2) # (B, T-1)
        exponents = torch.arange(generate_delta_latent.shape[1]) # 0, 1, ..., T-2
        decay_factors = torch.pow(self.decay_factor, exponents) # gamma^0, gamma^1, ..., gamma^(T-2)
        norm = torch.sum(norm * decay_factors[None, :], dim=1) # (B,)
        loss = torch.mean(norm)
        return loss
        