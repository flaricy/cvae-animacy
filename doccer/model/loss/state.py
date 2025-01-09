import torch 
import torch.nn as nn 
import omegaconf

class StateLoss(nn.Module):
    position_mask = [0, 1, 4, 5, 8, 9]
    velocity_mask = [2, 3, 6, 7, 10, 11]
    def __init__(self, cfg : omegaconf.dictconfig.DictConfig):
        super(StateLoss, self).__init__()
        self.position_weight = cfg.position_weight
        self.velocity_weight = cfg.velocity_weight
        if 'decay_factor_gamma' in cfg:
            self.decay_factor = cfg.decay_factor_gamma
        
    def forward(self, pred_state, gt_state, decay=False):
        '''
        state: (B, T, state_dim)
        '''

        if len(pred_state.shape) == 3: # with time dimension
            position_error = torch.sum((pred_state[:, :, StateLoss.position_mask] - gt_state[:, :, StateLoss.position_mask]) ** 2, dim=2) # (B, T)
            velocity_error = torch.sum((pred_state[:, :, StateLoss.velocity_mask] - gt_state[:, :, StateLoss.velocity_mask]) ** 2, dim=2) # (B, T)
            if decay:
                exponents = torch.arange(pred_state.shape[1], device=pred_state.device) # 0, 1, ..., T-1
                decay_factors = torch.pow(self.decay_factor, exponents) # gamma^0, gamma^1, ..., gamma^(T-1)
                position_error = position_error * decay_factors[None, :]
                velocity_error = velocity_error * decay_factors[None, :]

            position_loss = torch.mean(torch.sum(position_error, dim=1))
            velocity_loss = torch.mean(torch.sum(velocity_error, dim=1))
            return dict(
                position_loss=position_loss,
                velocity_loss=velocity_loss,
                sum_loss=position_loss * self.position_weight + velocity_loss * self.velocity_weight
            )

        elif len(pred_state.shape) == 2: # without time dimension
            position_error = torch.sum((pred_state[:, StateLoss.position_mask] - gt_state[:, StateLoss.position_mask]) ** 2, dim=1) # (B)
            velocity_error = torch.sum((pred_state[:, StateLoss.velocity_mask] - gt_state[:, StateLoss.velocity_mask]) ** 2, dim=1) # (B)

            position_loss = torch.mean(position_error)
            velocity_loss = torch.mean(velocity_error)
            return dict(
                position_loss=position_loss,
                velocity_loss=velocity_loss,
                sum_loss=position_loss * self.position_weight + velocity_loss * self.velocity_weight
            )