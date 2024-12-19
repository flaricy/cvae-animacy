from torch.utils.data import Dataset, DataLoader 
import omegaconf

def build_dataloader(cfg : omegaconf.dictconfig.DictConfig, dataset : Dataset):
    ret = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    return ret