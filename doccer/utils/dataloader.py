from torch.utils.data import Dataset, DataLoader 
import omegaconf

def build_dataloader(cfg : omegaconf.dictconfig.DictConfig, dataset : Dataset, shuffle : bool):
    ret = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=shuffle)
    return ret