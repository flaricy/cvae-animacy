import torch 
import torch.nn as nn 
import omegaconf
from ..model import build_model
from ..model.loss.state import StateLoss
from ..model.loss.kl_diverg import KLDivergLoss
from ..dataset.doccer import DoccerGTDataset, DoccerDynamicDataset
from ..utils.dataloader import build_dataloader
from ..utils.optimizer import build_optimizer 
from ..utils.scheduler import build_scheduler
from .process import TrajectoryCollector
from .simulator import Simulator
from tqdm import tqdm 

class Trainer(object):
    def __init__(self, cfg : omegaconf.dictconfig.DictConfig):
        self.cfg = cfg
        self._prepare_data()
        self._prepare_model()
        
    def train(self):
        kl_loss_factor=self.cfg.train.kl_divergence_loss.factor_beta
        for epoch in range(self.cfg.train.epochs):
            self.model.eval()
            pieces = self.trajectory_collector.collect_trajectory(self.model, self.cfg.train.collector.num_trajectories)
            self.dynamic_dataset.merge_pieces(pieces)

            self.model.train()
            
            self.dynamic_dataset.update_clips(self.cfg.train.update_world_model.clip_length)
            world_model_dataloader = build_dataloader(self.cfg.train.update_world_model.dataloader, self.dynamic_dataset)
            for batch_count, data in enumerate(world_model_dataloader, start=1):
                
                self.world_model_optimizer.zero_grad()
                self.vae_model_optimizer.zero_grad()
                
                for key in data:
                    data[key] = data[key].to(self.cfg.device)
                    
                generate_state = torch.zeros_like(data['state'], device=self.cfg.device)
                generate_state[:, 0] = data['state'][:, 0]
                for t in range(data['state'].shape[1] - 1):
                    generate_state[:, t + 1] = self.model.world_model(generate_state[:, t], data['action'][:, t])
                    
                loss = self.state_loss(generate_state, data['state'])
                loss.backward()
                self.world_model_optimizer.step()
                
                if batch_count == self.cfg.train.update_world_model.num_updates:
                    break
                
            self.dynamic_dataset.update_clips(self.cfg.train.update_policy_model.clip_length)
            policy_model_dataloader = build_dataloader(self.cfg.train.update_policy_model.dataloader, self.dynamic_dataset)
            for batch_count, data in enumerate(policy_model_dataloader, start=1):
                
                self.world_model_optimizer.zero_grad()
                self.vae_model_optimizer.zero_grad()
                
                for key in data:
                    data[key] = data[key].to(self.cfg.device)
                    
                generate_state = torch.zeros((data['gt_state'].shape[0], 1, data['gt_state'].shape[2]), device=self.cfg.device)
                generate_delta_latent = torch.zeros((data['gt_state'].shape[0], data['gt_state'].shape[1] - 1, self.cfg.model.posterior.output_dim[-1]), device=self.cfg.device)
                
                generate_state[:, 0] = data['gt_state'][:, 0]
                for t in range(data['gt_state'].shape[1] - 1):
                    latent_t, generate_delta_latent[:, t] = self.model.posterior(generate_state[:, -1], data['gt_state'][:, t + 1], return_delta=True)
                    action_t = self.model.policy(generate_state[:, -1], latent_t)
                    next_state = self.model.world_model(generate_state[:, -1], action_t)
                    generate_state = torch.cat([generate_state, next_state.unsqueeze(1)], dim=1)
                loss = self.state_loss(generate_state, data['gt_state']) + kl_loss_factor * self.kl_diver_loss(generate_delta_latent)
                loss.backward()
                self.vae_model_optimizer.step()
                
                if batch_count == self.cfg.train.update_policy_model.num_updates:
                    break
                
            self.world_model_scheduler.step()
            self.vae_model_scheduler.step()
                
            if epoch % self.cfg.train.kl_divergence_loss.beta_update_step == 0:
                kl_loss_factor *= self.cfg.train.kl_divergence_loss.beta_update_multiplier
            
            
    
    def _prepare_data(self):
        self.gt_dataset = DoccerGTDataset(self.cfg.dataset)
        self.dynamic_dataset = DoccerDynamicDataset(self.cfg.train.dynamic_dataset)
        self.simulator = Simulator(self.cfg.simulator)
        self.trajectory_collector = TrajectoryCollector(self.gt_dataset, self.cfg.device, self.simulator)
        
        
    def _prepare_model(self):
        self.model = build_model(self.cfg.model)
        self.model.to(self.cfg.device)
        
        self.world_model_optimizer = build_optimizer(self.cfg.train.optimizer, [p for p in self.model.world_model.parameters() if p.requires_grad])
        self.vae_model_optimizer = build_optimizer(
            self.cfg.train.optimizer, 
            [p for p_name, p in self.model.named_parameters() if p.requires_grad and 'world_model' not in p_name]
        )
        self.world_model_scheduler = build_scheduler(self.cfg.train.scheduler, self.world_model_optimizer)
        self.vae_model_scheduler = build_scheduler(self.cfg.train.scheduler, self.vae_model_optimizer)
        self.state_loss = StateLoss(self.cfg.train.state_loss)
        self.kl_diver_loss = KLDivergLoss(self.cfg.train.kl_divergence_loss)
        
        
        