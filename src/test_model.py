from re import T
import torch
import numpy as np
import os
import torch.optim as optim
torch.set_default_dtype(torch.float32)

from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import new_experiment, juicy_word, plot_losses, plot_equivariance_errors
from data import get_tvt_file_lists, get_test_file_lists,NumpyTimeSeriesDataset
from tqdm import tqdm
from models import get_model
from eval import (
    eval_test_set,
    plot_spectra,
    plot_spectrum,
    visualize_predictions_3D,
    mean_octahedral_equivariance_error,
    plot_equivariance,
    eval_equivariance
)
import argparse
import global_config as global_config
from transformations import apply_octahedral_augmentation, apply_so3_augmentation
import yaml
from box import Box

def _load_config_label(run_dir):
    setup_path = os.path.join(run_dir, 'experiment_setup.txt')
    if not os.path.exists(setup_path):
        return None, None
    with open(setup_path, 'r') as f:
        text = f.read()
    
    if 'Contents of' in text:
        yaml_part = text.split('Contents of', 1)[0].strip()
    else:
        yaml_part = text
    
    if 'Experiment config:' in yaml_part:
        yaml_part = yaml_part.split('Experiment config:', 1)[1].strip()
    
    cfg = yaml.safe_load(yaml_part)
    return Box(cfg)

def test_model_after_training(experiment_dir, device, compute_spectra=False):
    experiment_dir = experiment_dir
    config = _load_config_label(experiment_dir)
    print(f"WELCOME TO THE EXPERIMENT: {config.barcode}")

    torch.set_default_dtype(torch.float32)

    # Load the checkpoint
    checkpoint = torch.load(os.path.join(experiment_dir, f'{config.barcode}_checkpoint.pt'), map_location=device)

    # Extract the model state dict
    best_state = checkpoint["model"]

    # Initialize the model
    best_model = get_model(config.model).to(device)

    # Load the state dict
    #best_model.load_state_dict(best_state)
    best_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    # Set to evaluation mode
    best_model.eval()
    if config.loss_type == 'L1':
        criterion = F.l1_loss
    elif config.loss_type == 'L2':
        criterion = F.mse_loss

    # Use whole test set.
    config.dataset.test_samples = 'all'
    config.dataset.val_samples = 'all'
    print(f"default dtype: {torch.get_default_dtype()}")
    train_input_files, train_target_files, val_input_files, val_target_files, test_input_files, test_target_files = get_tvt_file_lists(config.dataset)

    train_dataset = NumpyTimeSeriesDataset(
        input_file_list=train_input_files,
        target_file_list=train_target_files,
        scaling_factors=None,
        augmentation_group=config.train_aug_group,
        scalar_predictor=config.scalar_predictor,
    )

    val_dataset = NumpyTimeSeriesDataset(
        input_file_list=val_input_files,
        target_file_list=val_target_files,
        scaling_factors=train_dataset.get_scaling_factors(),
        augmentation_group=config.val_aug_group,
        scalar_predictor=config.scalar_predictor,
    )

    test_dataset = NumpyTimeSeriesDataset(
        input_file_list=test_input_files,
        target_file_list=test_target_files,
        scaling_factors=train_dataset.get_scaling_factors(),
        augmentation_group=config.test_aug_group,
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

    if config.memorization_test:
        val_loader = train_loader
    else:   
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.eval_batch_size, 
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=8,
        )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.eval_batch_size, 
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )

    additional_test_loaders = {}
    if 'additional_test_datasets' in config.dataset:
        for name, params in config.dataset.additional_test_datasets.items():
            input_files, target_files = get_test_file_lists(params.input_prefix, params.target_prefix)
            dataset = NumpyTimeSeriesDataset(
                input_file_list=input_files,
                target_file_list=target_files,
                scaling_factors=train_dataset.get_scaling_factors(),
                augmentation_group=config.test_aug_group,
                scalar_predictor=config.scalar_predictor,
            )
            additional_test_loaders[name] = DataLoader(
                dataset, 
                batch_size=config.eval_batch_size, 
                num_workers=1,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=8,
            )

    config_tmp = config.copy()
    config_tmp.dataset.input_prefix = 'channel_middle_filtered_fs4'
    config_tmp.dataset.target_prefix = 'channel_middle'
    _, _, _, _, test_middle_input_files, test_middle_target_files = get_tvt_file_lists(config_tmp.dataset)
    
    test_dataset_middle = NumpyTimeSeriesDataset(
        input_file_list=test_middle_input_files,
        target_file_list=test_middle_target_files,
        scaling_factors=train_dataset.get_scaling_factors(),
        augmentation_group=config_tmp.test_aug_group,
        scalar_predictor=config_tmp.scalar_predictor,
    )
    test_loader_middle = DataLoader(
        test_dataset_middle, 
        batch_size=config_tmp.eval_batch_size, 
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )

    config_tmp.dataset.input_prefix = 'channel_nearwall_filtered_fs4'
    config_tmp.dataset.target_prefix = 'channel_nearwall'
    _, _, _, _, test_nearwall_input_files, test_nearwall_target_files = get_tvt_file_lists(config_tmp.dataset)
    test_dataset_nearwall = NumpyTimeSeriesDataset(
        input_file_list=test_nearwall_input_files,
        target_file_list=test_nearwall_target_files,
        scaling_factors=train_dataset.get_scaling_factors(),
        augmentation_group=config_tmp.test_aug_group,
        scalar_predictor=config_tmp.scalar_predictor,
    )
    test_loader_nearwall = DataLoader(
        test_dataset_nearwall, 
        batch_size=config_tmp.eval_batch_size, 
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )
    
    scaling_factors = {'input_std': train_dataset.get_scaling_factors()['input_std'].to(device),
                       'input_mean': train_dataset.get_scaling_factors()['input_mean'].to(device),
                       'target_std': train_dataset.get_scaling_factors()['target_std'].to(device),
                       'target_mean': train_dataset.get_scaling_factors()['target_mean'].to(device)}
    with torch.inference_mode():                    
        test_loss = eval_test_set(config, test_loader, best_model, criterion, scaling_factors)
        print(f"Test Loss: {test_loss:.4f}")
        k = None
        spec_error = None
        if config.task_type == 'sr':
            if compute_spectra:
                abs_equiv_error, (k, spec_error), abs_equiv_error_u, abs_equiv_error_v, abs_equiv_error_w = eval_equivariance(config, test_loader, best_model, compute_spectra=True, scaling_factors=scaling_factors)
                plot_spectrum(k, spec_error, os.path.join(experiment_dir, f'{config.barcode}_test_equiv_error_spectrum_after.png'),'Test error spectrum')
               
            else:
                print("Computing equivariance error only")
                abs_equiv_error, abs_equiv_error_u, abs_equiv_error_v, abs_equiv_error_w = eval_equivariance(config, test_loader, best_model, compute_spectra=False, scaling_factors=scaling_factors)

            additional_test_losses = dict.fromkeys(additional_test_loaders.keys())
            additional_abs_equiv_errors = dict.fromkeys(additional_test_loaders.keys())
            for name, loader in additional_test_loaders.items():
                additional_test_losses[name] = eval_test_set(config, loader, best_model, criterion, scaling_factors)
                print(f"Test {name} Loss: {additional_test_losses[name]:.4f}")
                additional_abs_equiv_errors[name] = eval_equivariance(config, loader, best_model, compute_spectra=False, scaling_factors=scaling_factors)
            
            # eval model on both nearwall and middle. one of these is a duplicate eval to the test eval above, but it's ok :)
            test_loss_middle = eval_test_set(config, test_loader_middle, best_model, criterion, scaling_factors)
            additional_abs_equiv_errors_middle = eval_equivariance(config, test_loader_middle, best_model, compute_spectra=False, scaling_factors=scaling_factors)
            print(f"Test Middle Loss: {test_loss_middle:.4f}")
            test_loss_nearwall = eval_test_set(config, test_loader_nearwall, best_model, criterion, scaling_factors)
            additional_abs_equiv_errors_nearwall = eval_equivariance(config, test_loader_nearwall, best_model, compute_spectra=False, scaling_factors=scaling_factors)
            print(f"Test Nearwall Loss: {test_loss_nearwall:.4f}")
            test_results = {        
                "test_loss": test_loss,
                "abs_equiv_error": abs_equiv_error,
                "abs_equiv_error_u": abs_equiv_error_u,
                "abs_equiv_error_v": abs_equiv_error_v,
                "abs_equiv_error_w": abs_equiv_error_w,
                "k_spec_error": k,
                "spec_error": spec_error,
                "additional_test_losses": additional_test_losses,
                "additional_abs_equiv_errors": additional_abs_equiv_errors,
                "test_loss_middle": test_loss_middle,
                "test_loss_nearwall": test_loss_nearwall,
                "additional_abs_equiv_errors_middle": additional_abs_equiv_errors_middle,
                "additional_abs_equiv_errors_nearwall": additional_abs_equiv_errors_nearwall,
            }

        else:
            abs_equiv_error = eval_equivariance(config, test_loader, best_model, compute_spectra=False, scaling_factors=scaling_factors) # Just returns the error when compute_spectra is False
            print(f"Test Abs Equiv Error: {abs_equiv_error:.4E}")
            test_results = {        
            "test_loss": test_loss,
            "abs_equiv_error": abs_equiv_error,
            }
        np.save(os.path.join(experiment_dir, f'{config.barcode}_test_results.npy'), test_results)

        visualize_predictions_3D(
            model=best_model,
            loader=train_loader,
            output_path=os.path.join(experiment_dir, f'{config.barcode}_train_predictions_bestmodel_after.png'),
            title='Train'
        )
        plot_equivariance(
            model=best_model,
            loader=train_loader,
            output_path=os.path.join(experiment_dir, f'{config.barcode}_train_equivariance_after.png'),
            title='Train'
        )

        visualize_predictions_3D(
            model=best_model,
            loader=val_loader,
            output_path=os.path.join(experiment_dir, f'{config.barcode}_val_predictions_bestmodel_after.png'),
            title='Val'
        )
        plot_equivariance(
            model=best_model,
            loader=val_loader,
            output_path=os.path.join(experiment_dir, f'{config.barcode}_val_equivariance_after.png'),
            title='Val'
        )

        for name, loader in additional_test_loaders.items():
            visualize_predictions_3D(
                model=best_model,
                loader=loader,
                output_path=os.path.join(experiment_dir, f'{config.barcode}_test_{name}_predictions_bestmodel_after.png'),
                title=name
            )
            plot_equivariance(
                model=best_model,
                loader=loader,
                output_path=os.path.join(experiment_dir, f'{config.barcode}_test_{name}_equivariance_after.png'),
                title=name
            )
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', type=str, help='Path to experiment directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--compute_spectra', action='store_true', help='compute equivariance error spectra')

    args = parser.parse_args()
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # required on macOS/Python 3.13
    
    test_model_after_training(args.experiment_dir, args.device, args.compute_spectra)