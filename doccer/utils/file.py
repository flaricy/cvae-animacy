from pathlib import Path
from datetime import datetime 

class FileHelper(object):
    def __init__(self, path:str, comment:str=None):
        self.parent_dir = Path(path)
        current_time = datetime.now()
        folder_name = current_time.strftime("%Y-%m-%d-%H-%M-%S")
        if comment is not None:
            folder_name += f'-{comment}'
        self.path = self.parent_dir / folder_name
        self.path.mkdir(parents=True, exist_ok=True)
        
        self._set_up()
        
    def _set_up(self):
        self.sub_dir = dict(
            log=self.path / 'log',
            ckpts=self.path / 'ckpts',
            config=self.path / 'config',
            vis=self.path / 'vis',
        )
        for key in self.sub_dir:
            self.sub_dir[key].mkdir(parents=True, exist_ok=True)
            
    def get_log_path(self):
        return str(self.sub_dir['log'])
    
    def get_ckpts_path(self):
        return str(self.sub_dir['ckpts'])
    
    def get_config_path(self):
        return str(self.sub_dir['config'])
    
    def get_vis_path(self):
        return str(self.sub_dir['vis'])
    
    def make_sub_vis_dir(self, epoch:int):
        tmp_path = self.sub_dir['vis'] / f"epoch_{epoch}"
        tmp_path.mkdir(parents=True, exist_ok=True)
        return str(tmp_path)
        
