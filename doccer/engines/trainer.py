import torch 
import torch.nn as nn 
import omegaconf
from ..model import build_model
from ..dataset import build_dataset
from ..utils.dataloader import build_dataloader
from ..utils.optimizer import build_optimizer 
from ..utils.scheduler import build_scheduler

class Trainer(object):
    def __init__(self, cfg : omegaconf.dictconfig.DictConfig):
        self.cfg = cfg
        self._prepare_data()
        self._prepare_model()
        
    def train(self):
        for epoch in range(self.cfg.train.epochs):
            self.model.train()
            for data in self.dataloader:
                pass
    
    def _prepare_data(self):
        gt_dataset = build_dataset(self.cfg.dataset)
        self.gt_dataloader = build_dataloader(self.cfg.train.gt_dataloader, gt_dataset, shuffle=True)
        
        
    def _prepare_model(self):
        self.model = build_model(self.cfg.model)
        self.model.to(self.cfg.device)
        
        self.optimizer = build_optimizer(self.cfg.train.optimizer, self.model)
        self.scheduler = build_scheduler(self.cfg.train.scheduler, self.optimizer)
        
        
        
        