import pickle 
import numpy as np
np.set_printoptions(threshold=np.inf)

with open("data/version3_20fps/2.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())
for key in data.keys():
    print(key, data[key].shape, data[key].dtype)
    
print("ACTION")
print(data['action'][:300])