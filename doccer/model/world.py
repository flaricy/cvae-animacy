import torch 
import torch.nn as nn 
from .builder import MODELS 
from .misc import get_act_module
from .distr import GaussianDistributor
import omegaconf

@MODELS.register_module()
class MLPWorldModel(nn.Module):
    def __init__(self, cfg : omegaconf.dictconfig.DictConfig):
        super(MLPWorldModel, self).__init__()
        self.layers = nn.Sequential()
        for i in range(len(cfg.dim) - 2):
            self.layers.append(nn.Linear(cfg.dim[i], cfg.dim[i + 1]))
            self.layers.append(get_act_module(cfg.act))
        self.layers.append(nn.Linear(cfg.dim[-2], cfg.dim[-1]))
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, state_t, action_t):
        '''
        state_t: [batch_size, state_dim]
        action_t: [batch_size, action_dim]
        '''
        x = torch.cat([state_t, action_t], dim=1) # [batch_size, state_dim + action_dim]
        return self.layers(x)

@MODELS.register_module()
class TransformerWorldModel(nn.Module):
    def __init__(self, cfg : omegaconf.dictconfig.DictConfig):
        super(TransformerWorldModel, self).__init__()
        self.embed_body = nn.Linear(4, cfg.hidden_dim)
        self.embed_action = nn.Linear(5, cfg.hidden_dim)

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=cfg.hidden_dim,
                nhead=cfg.nhead,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                activation=get_act_module(cfg.act),
                batch_first=True,
            ),
            num_layers=cfg.num_layers,
        )
        self.unpack_body = nn.Linear(cfg.hidden_dim, 4)
        self.unpack_action = nn.Linear(cfg.hidden_dim, 5)

    def forward(self, state_t, action_t):
        '''
        state_t: [batch_size, state_dim]
        action_t: [batch_size, action_dim]
        '''
        B, _ = state_t.shape
        state_t = state_t.reshape(B, 3, 4)
        action_t = action_t.reshape(B, 2, 5).float()
        state_embedding = self.embed_body(state_t) # [B, 3, hidden_dim]
        action_embedding = self.embed_action(action_t)
        x = torch.cat([state_embedding, action_embedding], dim=1) # [B, 5, hidden_dim]
        y = self.transformer(x)
        y_body = self.unpack_body(y[:, :3]) # [B, 3, 4]
        state_t_1 = y_body.reshape(B, -1) # [B, 12]
        return state_t_1