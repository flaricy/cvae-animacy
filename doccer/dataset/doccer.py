from torch.utils.data import Dataset
import omegaconf
from pathlib import Path 
import pickle
from tqdm import tqdm 
import torch

class DoccerGTDataset(Dataset):
    def __init__(self, cfg : omegaconf.dictconfig.DictConfig):
        self.clips = []
        self.datas = []
        for data_index, file in enumerate(tqdm(list(Path(cfg.path.raw_data_path).iterdir()), desc='Loading dataset')):
            if not file.is_file() or file.suffix != ".pkl":
                continue 
            with open(file, "rb") as f:
                data = pickle.load(f)
            cur_length = data['state'].shape[0]
            self.datas.append(data)
            
            if cur_length <= cfg.sample.max_length:
                self.clips.append((data_index, 0, cur_length - 1))
            else:
                for l in range(0, cur_length - cfg.sample.max_length + 1):
                    self.clips.append((data_index, l, l + cfg.sample.max_length - 1))

        self.clips = self.clips[::cfg.sample.downsample_rate]
        self.to_tensor = cfg.to_tensor
                    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        data_index, start, end = self.clips[idx]

        ret_dict = dict(
            state=self.datas[data_index]['state'][start : end + 1],
            action=self.datas[data_index]['action'][start : end + 1]
        )
        if self.to_tensor:
            for key in ret_dict:
                ret_dict[key] = torch.from_numpy(ret_dict[key])
        return ret_dict


class DoccerDynamicDataset(Dataset):
    def __init__(self, cfg : omegaconf.dictconfig.DictConfig):
        self.cfg = cfg 
        self.pieces = []
        '''
        piece:
            state: [T, D]
            action: [T, A]
            gt_state: [T, D]
        '''
        
    def merge_pieces(self, new_piece : list):
        if len(self.pieces) + len(new_piece) > self.cfg.max_num_trajectories:
            num_removals = len(self.pieces) + len(new_piece) - self.cfg.max_num_trajectories
            self.pieces = self.pieces[num_removals:]
        self.pieces.extend(new_piece)
        
    def update_clips(self, clip_length : int):
        self.clip_length = clip_length
        self.clips = []
        for piece_index, piece in enumerate(self.pieces):
            if piece['state'].shape[0] < clip_length:
                continue 
            for l in range(0, piece['state'].shape[0] - clip_length + 1):
                self.clips.append((piece_index, l))
                
    def __getitem__(self, index):
        piece_index, clip_start = self.clips[index]
        clip_end = clip_start + self.clip_length - 1
        return dict(
            state=self.pieces[piece_index]['state'][clip_start : clip_end + 1],
            action=self.pieces[piece_index]['action'][clip_start : clip_end + 1],
            gt_state=self.pieces[piece_index]['gt_state'][clip_start : clip_end + 1]
        )