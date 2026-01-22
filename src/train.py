import torch
import numpy as np
import os
import torch.optim as optim

from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import new_experiment, juicy_word, plot_losses
from data import get_tvt_file_lists, NumpyTimeSeriesDataset
from tqdm import tqdm
from models import get_model

import argparse
import global_config as global_config
from transformations import apply_octahedral_augmentation
from test_model import test_model_after_training

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment config file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--subdirectory', type=str, default=None, help='Subdirectory to save the experiment to, convenient for tracking sweeps')
    print(f"Default dtype: {torch.get_default_dtype()}")
    args = parser.parse_args()

    device = args.device
    
    config = new_experiment(args.config, args.subdirectory)
    print(f"WELCOME TO THE EXPERIMENT: {config.barcode}")

    print(f"Using seed: {config.seed}")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Model 
    model = get_model(config.model).to(device)
    print(f'Model param count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.loss_type == 'L1':
        criterion = F.l1_loss
    elif config.loss_type == 'L2':
        criterion = F.mse_loss

    # Train and val data loaders
    train_input_files, train_target_files, val_input_files, val_target_files, _, _ = get_tvt_file_lists(config.dataset)
    print(f"Number of train samples: {len(train_input_files)}")
    print(f"Number of val samples: {len(val_input_files)}")
    
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
        scaling_factors=train_dataset.scaling_factors,
        augmentation_group=config.val_aug_group,
        scalar_predictor=config.scalar_predictor,
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        num_workers=0,
        pin_memory=True,
        shuffle=True
    )

    if config.memorization_test:
        val_loader = train_loader

    else:   
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.eval_batch_size, 
            num_workers=0,
            pin_memory=True,
        )
        

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf') 
    best_epoch = 0

    for epoch in range(config.epochs):      
            
        model.train()
            
        running_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc="Training"):
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

        
        # Validation loop
        if epoch % config['validation_interval'] == 0:
            plot_losses(train_losses, val_losses, best_epoch, os.path.join(config['directory'], f'{config.barcode}_loss_plot.png'))
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for inputs, targets in tqdm(val_loader, desc="Validation loss"):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    pred = model(inputs)
                    val_loss += criterion(pred, targets).item()
                    
                val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch:4d}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Update best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        # Save model at either save interval or stopping criteria (which are early stopping + end of training)
        if  (epoch % config['save_interval'] == 0) or \
            (epoch > best_epoch + config.early_stopping_patience) or \
            (epoch == config.epochs - 1):

            print(f"----> Saving model from epoch {best_epoch} (val loss: {best_val_loss}). {juicy_word()}!")
            torch.save(
                {
                    "model": best_state_cpu,
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "seed": config.seed
                },
                os.path.join(config['directory'], f'{config.barcode}_checkpoint.pt'),
            )

        if epoch > best_epoch + config.early_stopping_patience:
            print(f"\n\n=== {juicy_word()}! {juicy_word()}! {juicy_word()}! EARLY STOPPING at epoch {epoch} (no improvement since {best_epoch}) ===\n")
            break


    del model
    torch.cuda.empty_cache()

    # Test after training
    if config.test_after_training:
        print(f"----> Running test eval on model from Epoch {best_epoch}")
        test_model_after_training(config['directory'], device)

    
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # required on macOS/Python 3.13
    main()
