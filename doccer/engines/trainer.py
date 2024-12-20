import torch 
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter
import omegaconf
import logging, copy
from ..model import build_model
from ..model.loss.state import StateLoss
from ..model.loss.kl_diverg import KLDivergLoss
from ..dataset.doccer import DoccerGTDataset, DoccerDynamicDataset
from ..utils.dataloader import build_dataloader
from ..utils.optimizer import build_optimizer 
from ..utils.scheduler import build_scheduler
from ..utils.misc import AverageHandler
from ..utils.file import FileHelper
from .process import TrajectoryCollector
from .simulator import Simulator
from tqdm import tqdm 

class Trainer(object):
    def __init__(self, cfg : omegaconf.dictconfig.DictConfig):
        self.cfg = cfg
        self._prepare_logging()
        self._prepare_data()
        self._prepare_model()
        
    def train(self):
        kl_loss_factor=self.cfg.train.kl_divergence_loss.factor_beta
        self.loss_handlers = dict(
            world_model=AverageHandler(),
            vae_state=AverageHandler(),
            kl_divergence=AverageHandler(),
        )
        self.writer = SummaryWriter(log_dir=self.file_helper.get_log_path())
        for epoch in range(self.cfg.train.epochs):
            if epoch % self.cfg.train.collector.collect_every_n_epochs == 0:
                logging.info(f"epoch {epoch} | Collecting simulated trajectories & updating dynamic dataset")
                self.model.eval()
                with torch.no_grad():
                    pieces = self.trajectory_collector.collect_trajectory(self.model, self.cfg.train.collector.num_trajectories)
                self.dynamic_dataset.merge_pieces(pieces)
                
                self.dynamic_dataset_for_world_model = copy.deepcopy(self.dynamic_dataset)
                self.dynamic_dataset_for_world_model.update_clips(self.cfg.train.update_world_model.clip_length)
                logging.info(f"epoch {epoch} | dataset length for world model: {len(self.dynamic_dataset_for_world_model)}")
                world_model_dataloader = build_dataloader(self.cfg.train.update_world_model.dataloader, self.dynamic_dataset_for_world_model)
                
                self.dynamic_dataset_for_policy_model = copy.deepcopy(self.dynamic_dataset)
                self.dynamic_dataset_for_policy_model.update_clips(self.cfg.train.update_policy_model.clip_length)
                logging.info(f"epoch {epoch} | dataset length for policy model: {len(self.dynamic_dataset_for_policy_model)}")
                policy_model_dataloader = build_dataloader(self.cfg.train.update_policy_model.dataloader, self.dynamic_dataset_for_policy_model)
            
            self.model.train()
            for key in self.loss_handlers:
                self.loss_handlers[key].reset()
            
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
                self.loss_handlers['world_model'].update(loss.item())
                self.world_model_optimizer.step()
                
                if batch_count == self.cfg.train.update_world_model.num_updates:
                    break
                
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
                state_loss = self.state_loss(generate_state, data['gt_state']) 
                kl_loss = kl_loss_factor * self.kl_diver_loss(generate_delta_latent)
                loss = state_loss + kl_loss
                loss.backward()
                self.loss_handlers['vae_state'].update(state_loss.item())
                self.loss_handlers['kl_divergence'].update(kl_loss.item())
                
                self.vae_model_optimizer.step()
                
                if batch_count == self.cfg.train.update_policy_model.num_updates:
                    break
               
            log_message = f"epoch {epoch} | " 
            for key in self.loss_handlers:
                log_message += f"{key} loss: {self.loss_handlers[key].get_average():.4f}\t"
                self.writer.add_scalar(f"{key}_loss", self.loss_handlers[key].get_average(), epoch)
            logging.info(log_message)
                
            self.world_model_scheduler.step()
            self.vae_model_scheduler.step()
                
            if epoch % self.cfg.train.kl_divergence_loss.beta_update_step == 0:
                kl_loss_factor *= self.cfg.train.kl_divergence_loss.beta_update_multiplier
                
            if epoch % self.cfg.train.save_every_n_epochs == 0:
                save_dict = dict(
                    epoch=epoch,
                    model=self.model.state_dict(),
                    world_model_optimizer=self.world_model_optimizer.state_dict(),
                    vae_model_optimizer=self.vae_model_optimizer.state_dict(),
                )
                torch.save(save_dict, self.file_helper.get_ckpts_path() + f'/epoch_{epoch}.pth')
            
            
    
    def _prepare_data(self):
        self.gt_dataset = DoccerGTDataset(self.cfg.dataset)
        logging.info(f"Ground truth dataset length: {len(self.gt_dataset)}")
        self.dynamic_dataset = DoccerDynamicDataset(self.cfg.train.dynamic_dataset)
        self.simulator = Simulator(self.cfg.simulator, self.cfg.dataset.sample.scaling)
        self.trajectory_collector = TrajectoryCollector(self.gt_dataset, self.cfg.device, self.simulator)
        
        
    def _prepare_model(self):
        self.model = build_model(self.cfg.model)
        self.model.to(self.cfg.device)
        logging.info(f"model built, total parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        
        self.world_model_optimizer = build_optimizer(self.cfg.train.optimizer, [p for p in self.model.world_model.parameters() if p.requires_grad])
        self.vae_model_optimizer = build_optimizer(
            self.cfg.train.optimizer, 
            [p for p_name, p in self.model.named_parameters() if p.requires_grad and 'world_model' not in p_name]
        )
        self.world_model_scheduler = build_scheduler(self.cfg.train.scheduler, self.world_model_optimizer)
        self.vae_model_scheduler = build_scheduler(self.cfg.train.scheduler, self.vae_model_optimizer)
        self.state_loss = StateLoss(self.cfg.train.state_loss)
        self.kl_diver_loss = KLDivergLoss(self.cfg.train.kl_divergence_loss)
        
    def _prepare_logging(self):
        self.file_helper = FileHelper(self.cfg.train.log_dir, comment=self.cfg.train.comment if 'comment' in self.cfg.train else None)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.file_helper.get_log_path() + '/app.log', mode='w')
            ]
        )