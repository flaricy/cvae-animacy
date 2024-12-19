import pickle 
import numpy as np 
from pathlib import Path 
from tqdm import tqdm

FROM_FPS = 60
TO_FPS = 30

def process_sequence(data):
    '''
    data:
        state: (T, 12)
        action: (T, 10)
    '''
    stride = FROM_FPS // TO_FPS
    if data['state'].shape[0] % stride != 0:
        rest = stride - data['state'].shape[0] % stride
        data['state'] = np.concatenate([data['state'], *([data['state'][-1:]] * rest)], axis=0)
        data['action'] = np.concatenate([data['action'], *([data['action'][-1:]] * rest)], axis=0)
    state = data['state'][::stride]
    action = np.zeros((data['action'].shape[0] // stride, data['action'].shape[1]), dtype=bool)
    for i in range(stride):
        action = action | data['action'][i::stride]
    return state, action

if __name__ == "__main__":
    original_path_dir = Path("data/version1_60fps")
    new_path_dir = Path("data/version1_30fps")
    new_path_dir.mkdir(parents=True, exist_ok=True)
    
    for file in tqdm(original_path_dir.iterdir()):
        if not file.is_file() or file.suffix != ".pkl":
            continue 
        with open(file, "rb") as f:
            data = pickle.load(f)
        state, action = process_sequence(data)
        new_data = dict(state=state, action=action)
        with open(str(new_path_dir / file.name), "wb") as f:
            pickle.dump(new_data, f)
