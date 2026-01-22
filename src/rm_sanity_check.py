# Goal of this script is to check if there are any duplicates in the training dataset.
# JHTB states that 4000 time steps are available, and I'm just double checking we got 4000 unique ones.

import numpy as np
from data import get_tvt_file_lists, NumpyTimeSeriesDataset
import argparse
from utils import new_experiment
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to experiment config file')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
args = parser.parse_args()
device = args.device

config = new_experiment(args.config)
config.dataset.sampling_method = 'sequential'
train_input_files, train_target_files, _, _, _, _ = get_tvt_file_lists(config.dataset)

train_dataset = NumpyTimeSeriesDataset(train_input_files, train_target_files, means_stds=None, augmentation_group=None)

print(f"Dataset length: {len(train_dataset)}")

values = []
for i, (inputs, targets) in enumerate(train_dataset):
    values.append(inputs.flatten().cpu().numpy())
    if i % 100 == 0:
        print(f"Processed {i} samples...")

print(f"Total samples: {len(values)}")
# Sanity check - when uncommented, below loop will throw error
# values = [values[0]] + values
#values.insert(50, values[0])

# Check for duplicates by comparing entire flattened arrays
for i in tqdm.tqdm(range(len(values))):
    for j in range(i+1, len(values)):
        if np.array_equal(values[i], values[j]):
            print(f"DUPLICATE: indices {i} and {j} are identical")