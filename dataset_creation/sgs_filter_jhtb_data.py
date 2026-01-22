import numpy as np
import os
import tqdm
import multiprocessing as mp
from functools import partial
import glob 
from filter_jhtb_data import box_filter_downsample



numpy_dir = '/data/NFS/potato/turbulence/numpy_4000'
input_prefix = 'channel_nearwall'
fs = 4

files = [f for f in glob.glob(os.path.join(numpy_dir, f"{input_prefix}*.npy")) 
         if not any(x in f for x in ["filtered", "sgs", "grad"])]
for file in tqdm.tqdm(files):
    data = np.load(file)
    filtered = box_filter_downsample(data, fs)
    
    # Compute SGS tensor (6 components: 11,22,33,12,13,23)
    sgs_full = np.array([box_filter_downsample(data[i:i+1]*data[j:j+1], fs)[0] - filtered[i]*filtered[j] 
                         for i,j in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]])
    
    # Extract interior points only for SGS to match gradient shape
    sgs = sgs_full[:, 1:-1, 1:-1, 1:-1]
    
    # Compute velocity gradients explicitly (9 components, interior points only)
    u, v, w = filtered[0], filtered[1], filtered[2]
    grad = np.array([
        np.gradient(u, axis=0)[1:-1,1:-1,1:-1],  # du/dx
        np.gradient(u, axis=1)[1:-1,1:-1,1:-1],  # du/dy
        np.gradient(u, axis=2)[1:-1,1:-1,1:-1],  # du/dz
        np.gradient(v, axis=0)[1:-1,1:-1,1:-1],  # dv/dx
        np.gradient(v, axis=1)[1:-1,1:-1,1:-1],  # dv/dy
        np.gradient(v, axis=2)[1:-1,1:-1,1:-1],  # dv/dz
        np.gradient(w, axis=0)[1:-1,1:-1,1:-1],  # dw/dx
        np.gradient(w, axis=1)[1:-1,1:-1,1:-1],  # dw/dy
        np.gradient(w, axis=2)[1:-1,1:-1,1:-1],  # dw/dz
    ])
    
    #np.save(file.replace(input_prefix, f'{input_prefix}_filtered_fs{fs}'), filtered)
    np.save(file.replace(input_prefix, f'{input_prefix}_sgs_fs{fs}'), sgs)
    np.save(file.replace(input_prefix, f'{input_prefix}_grad_fs{fs}'), grad)

