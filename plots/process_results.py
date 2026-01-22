import os
import re
from box import Box
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
# This script amalgamates the runs and saves the results to a csv file
FINAL_RESULTS_DIR_1BOX = '/home/rmcconke/implicit_data_augmentation/outputs/dec30_random'
FINAL_RESULTS_DIR_3BOX = '/home/rmcconke/implicit_data_augmentation/outputs/jan1_3box'

def find_run_dirs(exp_dir: str):
    subdirs = [os.path.join(exp_dir, d) for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    run_dirs = []
    for d in subdirs:
        files = os.listdir(d)
        if any(f.endswith('_test_results.npy') for f in files):
            run_dirs.append(d)
    return sorted(run_dirs)

def load_config(run_dir):
    setup_path = os.path.join(run_dir, 'experiment_setup.txt')
    with open(setup_path, 'r') as f:
        text = f.read()
    yaml_part = text.split('Contents of', 1)[0].strip()
    cfg = Box(yaml.safe_load(yaml_part))
    return cfg

if __name__ == "__main__":
    for FINAL_RESULTS_DIR in [FINAL_RESULTS_DIR_1BOX, FINAL_RESULTS_DIR_3BOX]:
        run_dirs = find_run_dirs(FINAL_RESULTS_DIR)
        print(f"Found {len(run_dirs)} runs in {FINAL_RESULTS_DIR}")

        results_list = []

        for run_dir in tqdm(run_dirs,desc="Processing runs...."):
            run_config = load_config(run_dir)
            run_results = np.load(os.path.join(run_dir, f"{run_config.barcode}_test_results.npy"), allow_pickle=True).item()
            flattened = pd.json_normalize(run_config, sep='_').to_dict(orient='records')[0]
            flattened.update(run_results)
            results_list.append(flattened)
        
        results = pd.DataFrame(results_list)
        results.to_csv(os.path.join(FINAL_RESULTS_DIR, 'final_results.csv'), index=False)