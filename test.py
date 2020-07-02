import numpy as np
import os

files = os.listdir()
for data_file in files:
    if not data_file.endswith(".npy"):
        continue
    res = 0
    print(data_file +"'s shape")
    with open(data_file, "rb") as f:
        res = np.load(f)
        while True:
            try:
                data = np.load(f)
            except:
                print(res.shape)
                break
            res = np.concatenate((res, data))