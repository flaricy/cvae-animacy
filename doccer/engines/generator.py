import torch
import numpy as np
import omegaconf
from ..model.builder import build_model
from ..dataset.doccer import DoccerGTDataset
from .simulator import Simulator

class Generator:
    def __init__(self, cfg:omegaconf.dictconfig.DictConfig, gen_ckpt:str):
        self.cfg = cfg
        self._prepare_dataset()
        self._prepare_model(gen_ckpt)
        self._prepare_simulator()

    def _prepare_model(self, gen_ckpt:str):
        self.gen_model = build_model(self.cfg.gen_model)
        self.gen_model.to(self.cfg.device)
        checkpoint = torch.load(gen_ckpt, map_location=self.cfg.device)
        self.gen_model.load_state_dict(checkpoint['model_state_dict'])

    def _prepare_dataset(self):
        self.dataset = DoccerGTDataset(self.cfg.gen_model_trainer.dataset)

    def _prepare_simulator(self):
        self.simulator = Simulator(self.cfg.simulator, state_scaling=self.cfg.gen_model_trainer.dataset.sample.scaling)
        self.simulator.initialize()

    def generate(self, initial_state, generate_length=80):
        '''
        initial_state: (state_D)
        '''
        self.gen_model.eval()
        with torch.no_grad():
            self.gen_model.reset()
            ret_state = np.zeros((generate_length, initial_state.shape[0]), dtype=np.float32)
            ret_state[0] = initial_state.numpy()
            for i in range(1, generate_length):
                action = self.gen_model.gen_forward(torch.from_numpy(ret_state[i - 1]).to(self.cfg.device))
                action = (action > 0).cpu().numpy()

                action = self.simulator.action_correction(action)
                result = self.simulator.conduct_action(action)
                if result is not None:
                    return ret_state[:i]

                ret_state[i] = self.simulator.get_state()

        return ret_state
