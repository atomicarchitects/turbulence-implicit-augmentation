import torch
import numpy as np
import os
from typing import Optional
from torch.utils.data import Dataset
from global_config import numpy_dir
from utils import get_timesteps_available, full_filename
import random
from transformations import apply_so3_augmentation, apply_octahedral_augmentation

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


class NumpyTimeSeriesDataset(Dataset):
    def __init__(
        self,
        input_file_list,
        target_file_list,
        scaling_factors = None,
        scale: bool = True,
        augmentation_group: Optional[str] = None,
        scalar_predictor: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        self.numpy_dir = numpy_dir
        self.inputs = []
        self.targets = []
        self.scalar_predictor = scalar_predictor
        self.scale_flag = scale

        for input_file, target_file in zip(input_file_list, target_file_list):
            input_numpy = np.load(os.path.join(numpy_dir, input_file)).astype(np.float32)
            target_numpy = np.load(os.path.join(numpy_dir, target_file)).astype(np.float32)
            self.inputs.append(input_numpy)
            self.targets.append(target_numpy)
            
        self.inputs = torch.from_numpy(np.array(self.inputs)).to(dtype)
        self.targets = torch.from_numpy(np.array(self.targets)).to(dtype)
    
        if scale:
            if 'sgs' in target_file_list[0]:
                self.scaling_factors = self.scale_tensor(scaling_factors)
            else:
                self.scaling_factors = self.scale_velocity(scaling_factors)

        if augmentation_group is not None:
            if augmentation_group == 'so3':
                self.inputs, self.targets = apply_so3_augmentation(self.inputs, self.targets)
            elif augmentation_group == 'oct':
                self.inputs, self.targets = apply_octahedral_augmentation(self.inputs, self.targets)

    def scale_velocity(self, scaling_factors):
        if scaling_factors is not None:
            self.input_mean = scaling_factors['input_mean']
            self.input_std = scaling_factors['input_std']
            self.target_mean = scaling_factors['target_mean']
            self.target_std = scaling_factors['target_std']

        else:
            self.input_mean = self.inputs.mean(dim=(0, 2, 3, 4), keepdim=True)
            self.input_std = self.inputs.std(dim=(0, 2, 3, 4), keepdim=True)
            self.target_mean = self.targets.mean(dim=(0, 2, 3, 4), keepdim=True)
            self.target_std = self.targets.std(dim=(0, 2, 3, 4), keepdim=True)
            
        self.inputs = (self.inputs - self.input_mean) / (self.input_std + 1e-8)
        self.targets = (self.targets - self.target_mean) / (self.target_std + 1e-8)
            
        return {
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'target_mean': self.target_mean,
            'target_std': self.target_std,
        }

    def scale_tensor(self, scaling_factors):
        if scaling_factors is not None:
            self.input_mean_magnitude = scaling_factors['input_mean_magnitude']
            self.target_mean_magnitude = scaling_factors['target_mean_magnitude']
            
        else:
            self.input_mean_magnitude = torch.linalg.norm(self.inputs, dim=1, keepdim=True).mean(dim=(0, 2, 3, 4), keepdim=True)
            self.target_mean_magnitude = torch.linalg.norm(self.targets, dim=1, keepdim=True).mean(dim=(0, 2, 3, 4), keepdim=True)
        
        self.inputs = self.inputs / (self.input_mean_magnitude + 1e-8)
        self.targets = self.targets / (self.target_mean_magnitude + 1e-8)
        return {
            'input_mean_magnitude': self.input_mean_magnitude,
            'target_mean_magnitude': self.target_mean_magnitude,
        }

    def get_scaling_factors(self):
        return self.scaling_factors

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        #input = torch.linalg.norm(input, dim=0, keepdim=True)
        if self.scalar_predictor:
            target = self.targets[idx]
            target = torch.linalg.norm(target, dim=0, keepdim=True)
        else:
            target = self.targets[idx]
        return input, target

def get_tvt_file_lists(dataset_config):
    timesteps_avail = get_timesteps_available(numpy_dir, dataset_config.input_prefix)
    print(f"Number of timesteps available: {len(timesteps_avail)}")
    t_train, t_val, t_test = tvt_split_list(timesteps_avail, dataset_config.train_split, dataset_config.val_split, dataset_config.test_split)
    if dataset_config.train_samples != 'all':
        if dataset_config.sampling_method == 'random':
            t_train = random.sample(t_train, dataset_config.train_samples)
        elif dataset_config.sampling_method == 'sequential':
            t_train = t_train[:dataset_config.train_samples]
    if dataset_config.val_samples != 'all':
        if dataset_config.sampling_method == 'random':
            t_val = random.sample(t_val, dataset_config.val_samples)
        elif dataset_config.sampling_method == 'sequential':
            t_val = t_val[:dataset_config.val_samples]
    if dataset_config.test_samples != 'all':
        if dataset_config.sampling_method == 'random':
            t_test = random.sample(t_test, dataset_config.test_samples)
        elif dataset_config.sampling_method == 'sequential':
            t_test = t_test[:dataset_config.test_samples]
    train_input_files = [full_filename(dataset_config.input_prefix, box_number, timestep) for box_number in dataset_config.boxes for timestep in t_train]
    train_target_files = [full_filename(dataset_config.target_prefix, box_number, timestep) for box_number in dataset_config.boxes for timestep in t_train]
    val_input_files = [full_filename(dataset_config.input_prefix, box_number, timestep) for box_number in dataset_config.boxes for timestep in t_val]
    val_target_files = [full_filename(dataset_config.target_prefix, box_number, timestep) for box_number in dataset_config.boxes for timestep in t_val]
    test_input_files = [full_filename(dataset_config.input_prefix, box_number, timestep) for box_number in dataset_config.boxes for timestep in t_test]
    test_target_files = [full_filename(dataset_config.target_prefix, box_number, timestep) for box_number in dataset_config.boxes for timestep in t_test]
    
    # assert all(os.path.exists(file) for file in train_input_files), f"Train file {train_input_files[0]} does not exist"
    # assert all(os.path.exists(file) for file in val_input_files), f"Val file {val_input_files[0]} does not exist"
    # assert all(os.path.exists(file) for file in test_input_files), f"Test file {test_input_files[0]} does not exist"
    # assert all(os.path.exists(file) for file in train_target_files), f"Train target file {train_target_files[0]} does not exist"
    # assert all(os.path.exists(file) for file in val_target_files), f"Val target file {val_target_files[0]} does not exist"
    # assert all(os.path.exists(file) for file in test_target_files), f"Test target file {test_target_files[0]} does not exist"
    return train_input_files, train_target_files, val_input_files, val_target_files, test_input_files, test_target_files



def tvt_split_list(list_to_split, train_split, val_split, test_split):
    n = len(list_to_split)
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)
    train_list = list_to_split[:train_end]
    val_list = list_to_split[train_end:val_end]
    test_list = list_to_split[val_end:]
    
    return train_list, val_list, test_list

def get_test_file_lists(input_prefix, target_prefix):
    timesteps_avail = get_timesteps_available(numpy_dir, input_prefix)
    print(f"Number of timesteps available: {len(timesteps_avail)}")
    test_input_files = [full_filename(input_prefix, box_number, timestep) for box_number in [0,1,2] for timestep in timesteps_avail]
    test_target_files = [full_filename(target_prefix, box_number, timestep) for box_number in [0,1,2] for timestep in timesteps_avail]
    return test_input_files, test_target_files