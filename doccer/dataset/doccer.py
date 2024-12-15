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
        self.data = dict()
        self.clips = []
        self.length = cfg.sample.length
        self.stride = cfg.sample.stride
        self.interval = cfg.sample.interval
        self.to_tensor = cfg.to_tensor
        for entries in tqdm(list(data_path.iterdir()), desc="Preparing dataset"):
            if entries.name.endswith(".pickle"):
                seq_name = entries.name.split(".")[0]
                with open(str(entries), "rb") as f:
                    cur_data = pickle.load(f)['data']
                    self.data[seq_name] = cur_data 
                    
                for l in range(0, self.data[seq_name]['ball'].shape[0] - (self.length - 1) * self.stride, self.interval):
                    self.clips.append((seq_name, l))
                    
        self.clips = self.clips[::cfg.sample.downsample_rate]
        
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        seq_name, start = self.clips[idx]
        end = start + (self.length - 1) * self.stride
        ret = dict()
        
        ret['ball'] = self.data[seq_name]['ball'][start : end + 1 : self.stride]
        ret['player_1'] = self.data[seq_name]['r1'][start : end + 1 : self.stride]
        ret['player_2'] = self.data[seq_name]['b1'][start : end + 1 : self.stride]
        
        if self.to_tensor:
            import torch 
            for key in ret:
                ret[key] = torch.from_numpy(ret[key])
                
        return ret
    
