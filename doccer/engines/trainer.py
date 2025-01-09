import shutil
import torch
import omegaconf
from ..model.builder import build_model
from ..model.loss.state import StateLoss
from ..dataset.doccer import DoccerGTDataset
from ..utils.dataloader import build_dataloader
from ..utils.optimizer import build_optimizer
from ..utils.scheduler import build_scheduler
from ..utils.file import FileHelper
from ..utils.misc import AverageHandler
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm

class Trainer:
    def __init__(self, cfg:omegaconf.dictconfig.DictConfig):
        self.cfg = cfg
        self._prepare_logging()

    def _prepare_logging(self):
        self.file_helper = FileHelper(self.cfg.world_model_trainer.log_dir, comment=self.cfg.world_model_trainer.comment if 'comment' in self.cfg.world_model_trainer else None)
        shutil.copy(self.cfg.config_path, self.file_helper.get_config_path())
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.file_helper.get_log_path() + '/app.log', mode='w'),
                logging.StreamHandler()
            ]
        )
        self.writer = SummaryWriter(log_dir=self.file_helper.get_log_path())


class WorldModelTrainer(Trainer):
    def __init__(self, cfg: omegaconf.dictconfig.DictConfig):
        super(WorldModelTrainer, self).__init__(cfg)
        self._prepare_data()
        self._prepare_model()
        self._prepare_optimizer()

    def _prepare_data(self):
        logging.info("Preparing data")
        dataset = DoccerGTDataset(self.cfg.world_model_trainer.dataset)
        logging.info(f"Dataset length: {len(dataset)}")
        self.dataloader = build_dataloader(self.cfg.world_model_trainer.dataloader, dataset)
        logging.info(f"Dataloader length: {len(self.dataloader)}")

    def _prepare_model(self):
        logging.info("Preparing model")
        self.model = build_model(cfg=self.cfg.world_model, use_omegaconf=True)
        self.model.to(self.cfg.device)

        self.state_loss = StateLoss(self.cfg.world_model_trainer.loss)

    def _prepare_optimizer(self):
        logging.info("Preparing optimizer")
        params = [p for p in self.model.parameters() if p.requires_grad]
        params_count = sum(p.numel() for p in params)
        logging.info(f"Number of parameters: {params_count}")
        self.optimizer = build_optimizer(self.cfg.world_model_trainer.optimizer, params)
        self.scheduler = build_scheduler(self.cfg.world_model_trainer.scheduler, self.optimizer)

    def train(self):
        logging.info("Start training")

        for epoch in tqdm(range(1, self.cfg.world_model_trainer.epochs + 1)):
            loss_logger = dict(
                position_loss=AverageHandler(),
                velocity_loss=AverageHandler(),
                sum_loss=AverageHandler(),
            )
            self.model.train()
            for batch_index, data in enumerate(self.dataloader):
                input_state = data['state'][:, 0].to(self.cfg.device) # (B, state_D)
                gt_output_state = data['state'][:, 1].to(self.cfg.device) # (B, state_D)
                action = data['action'][:, 0].to(self.cfg.device) # (B, action_D)

                model_output_state = self.model(input_state, action) # (B, state_D)

                loss_dict = self.state_loss(model_output_state, gt_output_state)
                for key in loss_dict.keys():
                    loss_logger[key].update(loss_dict[key].item())

                self.optimizer.zero_grad()
                loss_dict['sum_loss'].backward()
                self.optimizer.step()

            loss_message = f"Epoch: {epoch} "
            for key in loss_logger.keys():
                self.writer.add_scalar(key, loss_logger[key].get_average(), epoch)
                loss_message += f"{key}: {loss_logger[key].get_average():.4f} "

            logging.info(loss_message)


            self.scheduler.step()

class GenerationModelTrainer(Trainer):
    def __init__(self, cfg:omegaconf.dictconfig.DictConfig):
        super(GenerationModelTrainer, self).__init__(cfg)
        self._prepare_data()
        self._prepare_model()
        self._prepare_optimizer()

    def train(self):
        logging.info("Start training")
        for epoch in tqdm(range(1, self.cfg.gen_model_trainer.epochs + 1)):
            self.model.train()
            self.loss_logger = AverageHandler()
            for batch_index, data in enumerate(self.dataloader):
                state = data['state'].to(self.cfg.device) # (B, T, state_D)
                gt_action = data['action'].to(self.cfg.device).float() # (B, T, action_D)
                model_output_action = self.model(state) # (B, T, action_D)

                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    input=model_output_action,
                    target=gt_action,
                    reduction='mean'
                )

                self.loss_logger.update(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            self.writer.add_scalar('loss', self.loss_logger.get_average(), epoch)
            logging.info(f"Epoch: {epoch} Loss: {self.loss_logger.get_average():.4f}")


    def _prepare_data(self):
        logging.info("Preparing data")
        dataset = DoccerGTDataset(self.cfg.gen_model_trainer.dataset)
        logging.info(f"Dataset length: {len(dataset)}")
        self.dataloader = build_dataloader(self.cfg.gen_model_trainer.dataloader, dataset)
        logging.info(f"Dataloader length: {len(self.dataloader)}")

    def _prepare_model(self):
        logging.info("Preparing model")
        self.model = build_model(cfg=self.cfg.gen_model, use_omegaconf=True)
        self.model.to(self.cfg.device)

    def _prepare_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        params_count = sum(p.numel() for p in params)
        logging.info(f"Number of parameters: {params_count}")
        self.optimizer = build_optimizer(self.cfg.gen_model_trainer.optimizer, params)
        self.scheduler = build_scheduler(self.cfg.gen_model_trainer.scheduler, self.optimizer)