import torch
import torch.nn as nn
from .builder import MODELS
import omegaconf

@MODELS.register_module()
class LSTMModel(nn.Module):
    def __init__(self, cfg:omegaconf.dictconfig.DictConfig):
        super(LSTMModel, self).__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            batch_first=True,
            proj_size=cfg.proj_size
        )

        self.h0 = nn.Parameter(torch.zeros(1, cfg.proj_size))
        self.c0 = nn.Parameter(torch.zeros(1, cfg.hidden_size))

    def forward(self, state):
        '''
        state: (B, T, state_D)
        '''
        B, T, _ = state.shape
        h0 = torch.tile(self.h0.unsqueeze(1), (1, B, 1)) # (1, B, action_D)
        c0 = torch.tile(self.c0.unsqueeze(1), (1, B, 1)) # (1, B, hidden_size)
        output, _ = self.lstm(state, (h0, c0)) # output: (B, T, action_D)
        return output

    def reset(self):
        self.cur_h = self.h0.detach().clone() # (1, action_D)
        self.cur_c = self.c0.detach().clone()

    def gen_forward(self, new_state):
        '''
        new_state: (state_D)
        '''
        x = new_state.unsqueeze(0) # (1, state_D)
        output, (self.cur_h, self.cur_c) = self.lstm(x) # output: (1, action_D)
        return output.squeeze(0) # (action_D)