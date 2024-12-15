import torch

def gt_to_state(gt_data : dict):
    '''
    gt_data:
        ball: (B, T, 2)
        player_1: (B, T, 2)
        player_2: (B, T, 2)
    '''
    ret_state = torch.cat([gt_data['ball'], gt_data['player_1'], gt_data['player_2']], dim=-1) # (B, T, 6)
    return ret_state
    