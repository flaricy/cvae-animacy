from pathlib import Path 

class FileHelper:
    def __init__(self, data_dir:str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cur_file_path(self):
        max_id = -1
        for entry in self.data_dir.iterdir():
            if not entry.is_file():
                continue 
            if entry.suffix != '.pkl':
                continue 
            
            file_name = entry.name
            file_id = file_name.split('.')[0]
            max_id = max(max_id, int(file_id))
        
        file_name = str(max_id + 1) + '.pkl'
        return str(self.data_dir / file_name)
    
    
