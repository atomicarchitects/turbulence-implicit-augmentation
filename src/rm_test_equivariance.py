import torch
import numpy as np
import os
import torch.optim as optim

from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import new_experiment, juicy_word, plot_losses, plot_equivariance_errors
from data import get_tvt_file_lists, get_test_file_lists,NumpyTimeSeriesDataset
from tqdm import tqdm
from models import get_model
from eval import (
    eval_test_set,
    plot_spectra,
    visualize_predictions_3D,
    mean_octahedral_equivariance_error,
)
import argparse
import global_config as global_config
from transformations import apply_octahedral_augmentation, apply_so3_augmentation
from box import Box

from transformations import rotate_3d, get_all_octahedral_angles
model_config = {  "model_type": "sgsECNN",
  "is_3d": True,
  "continuous_group": False,
  "kernel_size": 3,
  "group": "so3",
  "num_radial_basis": 32,
  "lmax": 2, 
  "input_irreps": "0e + 1o + 2e",
  "hidden_irreps": [
    "15x0e + 15x1o + 15x2e",
  ],
  "output_irreps": "0e + 2e"
}
model_config = Box(model_config)


torch.set_default_dtype(torch.float32)
model = get_model(model_config)

for i in range(10):
    x = torch.randn(1, 9, 14, 14, 14)
    x_rot = rotate_3d(x, get_all_octahedral_angles()[17])
    y = model(x)
    y_rot_then_pred = model(x_rot)
    y_pred_then_rot = rotate_3d(y, get_all_octahedral_angles()[17])

    print(torch.allclose(y_rot_then_pred, y_pred_then_rot, rtol=1e-5, atol=1e-5))

torch.set_default_dtype(torch.float64)
model = get_model(model_config)
for i in range(10):
    x = torch.randn(1, 9, 14, 14, 14)
    x_rot = rotate_3d(x, get_all_octahedral_angles()[17])
    y = model(x)
    y_rot_then_pred = model(x_rot)
    y_pred_then_rot = rotate_3d(y, get_all_octahedral_angles()[17])

    print(torch.allclose(y_rot_then_pred, y_pred_then_rot, rtol=1e-5, atol=1e-5))













