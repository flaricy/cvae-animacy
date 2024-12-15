from collections import abc 
import numpy as np
import torch 

def is_seq_of(seq, expected_type, seq_type=None):
    if seq_type is None:
        exp_seq_type = abc.Sequence 
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
        
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True