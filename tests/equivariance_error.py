"""
Standalone test script for octahedral equivariance error computation.

This script tests whether a model exhibits equivariance to octahedral symmetry
transformations by comparing predictions on transformed inputs.
"""

import torch
import numpy as np
import argparse
import os
from torch.utils.data import DataLoader

# Import required modules from the project
import sys
sys.path.append("./src/")

from models import get_model
from data import NumpyTimeSeriesDataset, get_tvt_file_lists
from eval import octahedral_equivariance_error
from utils import new_experiment
import global_config
from transformations import rotate_octahedral_exact, gradient_channels_to_matrix


def load_trained_model(model, checkpoint_path, device='cuda:0'):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def test_equivariance_on_dataset(model, data_loader, dtype, device='cuda:0', num_batches=None):
    """
    Test equivariance error across a dataset.
    
    Args:
        model: The neural network model to test
        data_loader: DataLoader providing input-target pairs
        device: Device to run computations on
        num_batches: Number of batches to test (None = all batches)
    
    Returns:
        dict with mean absolute and relative equivariance errors
    """
    model.eval()
    
    abs_errors = []
    rel_errors = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if num_batches is not None and i >= num_batches:
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)

            inputs = inputs.to(dtype)
            targets = targets.to(dtype)
    
            # Get predictions
            pred = model(inputs)
            
            # Compute equivariance error
            abs_errors_batch, rel_errors_batch = octahedral_equivariance_error(
                inputs, pred, targets, model, batch_size=inputs.shape[0]
            )
            
            abs_errors.append(abs_errors_batch.cpu().numpy())
            rel_errors.append(rel_errors_batch.cpu().numpy())
            
    
    abs_errors = np.concatenate(abs_errors)
    rel_errors = np.concatenate(rel_errors)
    
    
    return {
        'mean_abs_error': np.mean(abs_errors),
        'mean_rel_error': np.mean(rel_errors),
        'std_abs_error': np.std(abs_errors),
        'std_rel_error': np.std(rel_errors),
        'max_abs_error': np.max(abs_errors),
        'max_rel_error': np.max(rel_errors),
        'min_abs_error': np.min(abs_errors),
        'min_rel_error': np.min(rel_errors),
        'all_abs_errors': abs_errors,
        'all_rel_errors': rel_errors
    }


def test_scalar_rotations(dtype=torch.float64):

    field = torch.tensor(
        [[[
            [[1.,2.,3.],
             [4.,5.,6.],
             [7.,8.,9.]],

            torch.zeros(3,3),
            torch.zeros(3,3),
        ]]], dtype=dtype
    )

    # ------ Ry(π/2) ------
    target_y = torch.tensor(
        [[[
            [[0.,0.,1.],
             [0.,0.,4.],
             [0.,0.,7.]],

            [[0.,0.,2.],
             [0.,0.,5.],
             [0.,0.,8.]],

            [[0.,0.,3.],
             [0.,0.,6.],
             [0.,0.,9.]],
        ]]], dtype=dtype
    )
    out_y = rotate_octahedral_exact(field, torch.tensor([[np.pi/2, 0., 0.]]), rotate_channels=False)
    assert torch.allclose(out_y, target_y, atol=1e-6)

    # ------ Rx(π/2) ------
    target_x = torch.tensor(
        [[[
            [[3.,6.,9.],
             [2.,5.,8.],
             [1.,4.,7.]],

            torch.zeros(3,3),
            torch.zeros(3,3),
        ]]], dtype=dtype
    )
    out_x = rotate_octahedral_exact(field, torch.tensor([[0., np.pi/2, 0.]]), rotate_channels=False)
    assert torch.allclose(out_x, target_x, atol=1e-6)

    # ------ Rz(-π/2) ------
    target_z = torch.tensor(
        [[[
            [[0.,0.,0.],
             [0.,0.,0.],
             [1.,2.,3.]],

            [[0.,0.,0.],
             [0.,0.,0.],
             [4.,5.,6.]],

            [[0.,0.,0.],
             [0.,0.,0.],
             [7.,8.,9.]],
        ]]], dtype=dtype
    )
    out_z = rotate_octahedral_exact(field, torch.tensor([[-np.pi/2, -np.pi/2, np.pi/2]]), rotate_channels=False)
    assert torch.allclose(out_z, target_z, atol=1e-6)

    print("Scalar rotation tests passed")



def test_vector_rotations(dtype=torch.float64):

    base = torch.tensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]], dtype=dtype)
    field = torch.zeros(1,3,3,3,3,dtype=dtype)
    field[0,0] = torch.stack([base, torch.zeros_like(base), torch.zeros_like(base)])

    # --- expected patterns ---
    pat_y = torch.tensor(
        [[[0.,0.,-1.],[0.,0.,-4.],[0.,0.,-7]],
         [[0.,0.,-2.],[0.,0.,-5.],[0.,0.,-8]],
         [[0.,0.,-3.],[0.,0.,-6.],[0.,0.,-9]]], dtype=dtype
    )
    target_y = torch.zeros_like(field)
    target_y[0,2] = pat_y

    pat_x = torch.tensor([[3.,6.,9.],[2.,5.,8.],[1.,4.,7.]], dtype=dtype)
    target_x = torch.zeros_like(field)
    target_x[0,0,0] = pat_x

    target_z = torch.zeros_like(field)
    target_z[0,1] = torch.stack([
        torch.tensor([[0.,0.,0.],[0.,0.,0.],[-1.,-2.,-3.]],dtype=dtype),
        torch.tensor([[0.,0.,0.],[0.,0.,0.],[-4.,-5.,-6.]],dtype=dtype),
        torch.tensor([[0.,0.,0.],[0.,0.,0.],[-7.,-8.,-9.]],dtype=dtype),
    ])

    # --- apply rotations ---
    out_y = rotate_octahedral_exact(field, torch.tensor([[np.pi/2,0.,0.]]), rotate_channels=True)
    out_x = rotate_octahedral_exact(field, torch.tensor([[0.,np.pi/2,0.]]), rotate_channels=True)
    out_z = rotate_octahedral_exact(field, torch.tensor([[-np.pi/2,-np.pi/2,np.pi/2]]), rotate_channels=True)

    assert torch.allclose(out_y, target_y, atol=1e-6)
    assert torch.allclose(out_x, target_x, atol=1e-6)
    assert torch.allclose(out_z, target_z, atol=1e-6)

    print("Vector rotation tests passed")


def test_tensor_rotations(dtype=torch.float64):

    base = torch.tensor([
        [[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]],
        torch.zeros(3,3),
        torch.zeros(3,3)
    ], dtype=dtype)

    pat_y = torch.tensor([
        [[0.,0.,1.],[0.,0.,4.],[0.,0.,7]],
        [[0.,0.,2.],[0.,0.,5.],[0.,0.,8]],
        [[0.,0.,3.],[0.,0.,6.],[0.,0.,9]],
    ], dtype=dtype)

    pat_x = torch.tensor([
        [[3.,6.,9.],[2.,5.,8.],[1.,4.,7]],
        torch.zeros(3,3),
        torch.zeros(3,3),
    ], dtype=dtype)

    pat_z = torch.tensor([
        [[0.,0.,0.],[0.,0.,0.],[1.,2.,3.]],
        [[0.,0.,0.],[0.,0.,0.],[4.,5.,6.]],
        [[0.,0.,0.],[0.,0.,0.],[7.,8.,9.]],
    ], dtype=dtype)

    # channel mixing from R T Rᵀ
    mix_y = {0:(8,1),1:(7,-1),2:(6,-1),3:(5,-1),4:(4,1),5:(3,1),6:(2,-1),7:(1,1),8:(0,1)}
    mix_x = {0:(0,1),1:(2,1),2:(1,-1),3:(6,1),4:(8,1),5:(7,-1),6:(3,-1),7:(5,-1),8:(4,1)}
    mix_z = {0:(4,1),1:(3,-1),2:(5,-1),3:(1,-1),4:(0,1),5:(2,1),6:(7,-1),7:(6,1),8:(8,1)}

    for ch in range(9):
        field = torch.zeros(1,9,3,3,3,dtype=dtype)
        field[0,ch] = base

        # --- build targets ---
        target_y = torch.zeros_like(field)
        c,sgn = mix_y[ch]; target_y[0,c] = sgn*pat_y

        target_x = torch.zeros_like(field)
        c,sgn = mix_x[ch]; target_x[0,c] = sgn*pat_x

        target_z = torch.zeros_like(field)
        c,sgn = mix_z[ch]; target_z[0,c] = sgn*pat_z

        out_y = rotate_octahedral_exact(field, torch.tensor([[np.pi/2,0.,0.]]), rotate_channels=True)
        out_x = rotate_octahedral_exact(field, torch.tensor([[0.,np.pi/2,0.]]), rotate_channels=True)
        out_z = rotate_octahedral_exact(field, torch.tensor([[-np.pi/2,-np.pi/2,np.pi/2]]), rotate_channels=True)

        assert torch.allclose(out_y, target_y, atol=1e-6)
        assert torch.allclose(out_x, target_x, atol=1e-6)
        assert torch.allclose(out_z, target_z, atol=1e-6)

    print("Rank-2 tensor rotation tests passed")




def main():
    parser = argparse.ArgumentParser(description='Test octahedral equivariance of a trained model')
    parser.add_argument('config', type=str, help='Path to experiment config file')
    parser.add_argument('--dtype', type=str, default='float64', help='Data type for computations (default: float64)', required=True)
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint (.pt file)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--num_batches', type=int, default=None, 
                        help='Number of batches to test (default: all)')
    parser.add_argument('--dataset_split', type=str, default='test', 
                        choices=['train', 'val', 'test'], 
                        help='Which dataset split to test on')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for testing')
    
    args = parser.parse_args()
    
    # Load config
    config = new_experiment(args.config)
    print(f"Testing equivariance for experiment: {config.barcode}")
    
    dtype = {
        'float32': torch.float32,
        'float64': torch.float64,
    }[args.dtype]

    test_scalar_rotations(dtype=dtype)
    test_vector_rotations(dtype=dtype)
    test_tensor_rotations(dtype=dtype)

    # Load model
    model = get_model(config.model).to(args.device, dtype=dtype)
    if args.checkpoint is None:
        print("No checkpoint provided, initializing model with random weights")
    else:
        print(f"Loading model from: {args.checkpoint}")
        model = load_trained_model(model, args.checkpoint, device=args.device)
        print(f"Model loaded successfully")
    
    # Print dtypes of model parameters
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, dtype: {param.dtype}")
    # Prepare dataset
    train_input_files, train_target_files, val_input_files, val_target_files, test_input_files, test_target_files = get_tvt_file_lists(config.dataset)
    
    if args.dataset_split == 'train':
        input_files, target_files = train_input_files, train_target_files
    elif args.dataset_split == 'val':
        input_files, target_files = val_input_files, val_target_files
    else:
        input_files, target_files = test_input_files, test_target_files
    
    # Get normalization stats from training data
    train_dataset = NumpyTimeSeriesDataset(
        input_file_list=train_input_files,
        target_file_list=train_target_files,
        augmentation_group=None,
    )
    
    test_dataset = NumpyTimeSeriesDataset(
        input_file_list=input_files,
        target_file_list=target_files,
        augmentation_group=None,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    print(f"\nTesting on {args.dataset_split} split with {len(test_dataset)} samples")
    print(f"Batch size: {args.batch_size}")
    if args.num_batches:
        print(f"Testing on first {args.num_batches} batches")
    print("-" * 80)
    
    # Run equivariance test
    results = test_equivariance_on_dataset(
        model, 
        test_loader, 
        dtype=dtype,
        device=args.device, 
        num_batches=args.num_batches
    )
    
    # Print results
    print("\nEquivariance Test Results:")
    print(f"Mean Absolute Equivariance Error: {results['mean_abs_error']:.6f} ± {results['std_abs_error']:.6f}")
    print(f"Mean Relative Equivariance Error: {results['mean_rel_error']:.6f} ± {results['std_rel_error']:.6f}")
    print()
    print(f"Max Absolute Equivariance Error: {results['max_abs_error']:.6f}")
    print(f"Max Relative Equivariance Error: {results['max_rel_error']:.6f}")
    print()
    print(f"Min Absolute Equivariance Error: {results['min_abs_error']:.6f}")
    print(f"Min Relative Equivariance Error: {results['min_rel_error']:.6f}")

if __name__ == "__main__":
    main()