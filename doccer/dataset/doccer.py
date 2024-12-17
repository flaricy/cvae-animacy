from torch.utils.data import Dataset
import omegaconf
from pathlib import Path 
import pickle
from tqdm import tqdm 
from .builder import DATASETS

@DATASETS.register_module()
class DoccerDataset(Dataset):
    def __init__(self, cfg: omegaconf.dictconfig.DictConfig):
        data_path = Path(cfg.path.raw_data_path)
        self.data = []
        self.length = cfg.sample.length
        self.stride = cfg.sample.stride
        self.interval = cfg.sample.interval
        self.to_tensor = cfg.to_tensor
        for seq_index, entries in enumerate(tqdm(list(data_path.iterdir()), desc="Preparing dataset")):
            if entries.name.endswith(".pickle"):
                seq_name = entries.name.split(".")[0]
                with open(str(entries), "rb") as f:
                    cur_data = pickle.load(f)['data']
                    self.data.append(cur_data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ret = dict()
        ret['ball'] = self.data[idx]['ball']
        ret['player_1'] = self.data[idx]['r1']
        ret['player_2'] = self.data[idx]['b1']
        return ret 
