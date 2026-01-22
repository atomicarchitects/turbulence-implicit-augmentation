from typing import Any


import torch
import numpy as np
import os
import torch.optim as optim
import sys
sys.path.append("./src/")
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
    octahedral_equivariance_error,
    visualize_equivariance,
    plot_equivariance,
)
import argparse
import global_config as global_config
from transformations import apply_octahedral_augmentation, apply_so3_augmentation


def test_model_equivariance(config_file, device='cuda:1'):
    config = new_experiment(config_file)

    print(f"Using seed: {config.seed}")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = get_model(config.model).to(device)
    if config.loss_type == 'L1':
        criterion = F.l1_loss
    elif config.loss_type == 'L2':
        criterion = F.mse_loss

    optimizer = optim.Adam(model.parameters())

    print(f'Model param count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    train_input_files, train_target_files, val_input_files, val_target_files, test_input_files, test_target_files = get_tvt_file_lists(config.dataset)

    train_dataset = NumpyTimeSeriesDataset(
        input_file_list=train_input_files,
        target_file_list=train_target_files,
        scaling_factors=None,
        augmentation_group=config.train_aug_group,
        scalar_predictor=config.scalar_predictor,
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
        shuffle=True
    )


    # Equivariance error at initialization
    input_example, label_example = next(iter(train_loader))
    input_example = input_example.to(device)
    label_example = label_example.to(device)
    abs_equiv_error, rel_equiv_error = octahedral_equivariance_error(
        input_example, model(input_example), label_example, model, batch_size=1
    )
    print(f"Mean equiv error: {torch.mean(abs_equiv_error)}")
    print(f"Max equiv error: {torch.max(abs_equiv_error.flatten())}")
    print(f"Mean rel equiv error: {torch.mean(rel_equiv_error)}")
    print(f"Max rel equiv error: {torch.max(rel_equiv_error.flatten())}")
    assert torch.max(abs_equiv_error.flatten()) < ABS_EQUIV_ERROR_THRESHOLD, f"Equiv error: {torch.max(abs_equiv_error.flatten())}"

    visualize_equivariance(input_example, model(input_example), label_example, model)
    plot_equivariance(input_example, model(input_example), label_example, model, output_path=f"equivariance_initial.png")
    #assert torch.max(abs_equiv_error.flatten()) < equiv_error_threshold, f"Equiv error: {torch.max(abs_equiv_error.flatten()) }"
    #assert torch.max(rel_equiv_error.flatten()) < equiv_error_threshold, f"Rel equiv error: {torch.max(rel_equiv_error.flatten())}"

    
    epoch_start = 0
    train_losses = []
    val_losses = []
    val_equiv_errors = []
    val_rel_equiv_errors = []
    best_val_loss = float('inf') 
    best_model = None
    best_epoch = 0
    val_loss = 0.0
    equiv_err = 0
    rel_equiv_err = 0
    is_last_epoch = False
    for epoch in tqdm(range(config.epochs), desc="Training"):


        learning_rate = config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate          
            
        model.train()
            
        running_loss = 0.0
            
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if config.train_aug_group is not None:
                if config.train_aug_group == "oct":
                    inputs, targets = apply_octahedral_augmentation(inputs, targets)

            pred = model(inputs)
            loss = criterion(pred, targets)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = running_loss / len(train_loader)
        if epoch == 0:
            print(f"Epoch {epoch}, Train Loss: {avg_train_loss}")
    print(f"Epoch {epoch}, Train Loss: {avg_train_loss}")
    print(f"After training")

    abs_equiv_error, rel_equiv_error = octahedral_equivariance_error(
        input_example, model(input_example), label_example, model, batch_size=1
    )
    print(f"Mean equiv error: {torch.mean(abs_equiv_error)}")
    print(f"Max equiv error: {torch.max(abs_equiv_error.flatten())}")
    print(f"Mean rel equiv error: {torch.mean(rel_equiv_error)}")
    print(f"Max rel equiv error: {torch.max(rel_equiv_error.flatten())}")

    # Equivariance error at end of training

    visualize_equivariance(input_example, model(input_example), label_example, model)

    abs_equiv_error = mean_octahedral_equivariance_error(input_example, model(input_example), label_example, model)[0].item()
    print(f"Equiv error at end of training: {abs_equiv_error}")
    assert torch.max(abs_equiv_error.flatten()) < ABS_EQUIV_ERROR_THRESHOLD, f"Equiv error: {torch.max(abs_equiv_error.flatten())}"


DEVICE = 'cuda:1'
ABS_EQUIV_ERROR_THRESHOLD = 1e-12
#test_model_equivariance("tests/ecnn_scalar_to_scalar_equivariant.yaml", DEVICE)
test_model_equivariance("tests/ecnn_vector_to_vector_equivariant.yaml", DEVICE)

test_model_equivariance("tests/ecnn_sgs_equivariant.yaml", DEVICE)
test_model_equivariance("tests/ecnn_2nd_to_scalar_equivariant.yaml", DEVICE)
