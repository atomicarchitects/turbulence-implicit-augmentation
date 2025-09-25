import torch
import numpy as np
import os
import torch.optim as optim

from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import new_experiment, juicy_word, plot_losses, plot_equivariance_errors
from data import get_tvt_file_lists, NumpyTimeSeriesDataset
from tqdm import tqdm
from models import get_model
from eval import plot_spectra, visualize_predictions_3D, mean_octahedral_equivariance_error, mean_so3_equivariance_error
import argparse
import global_config as global_config
from transformations import apply_octahedral_augmentation, apply_so3_augmentation
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment config file')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')
    args = parser.parse_args()
    device = args.device

    config = new_experiment(args.config)
    print(f"WELCOME TO THE EXPERIMENT: {config.barcode}")

    if config.seed == 'random':
        seed = np.random.default_rng().integers(0, 2**32 - 1)
    elif isinstance(config.seed, int):
        seed = config.seed
    else:
        raise ValueError(f"Invalid seed: {config.seed}")

    print(f"Using seed: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    model = get_model(config.model).to(device)  

    if config.loss_type == 'L1':
        criterion = F.l1_loss
    elif config.loss_type == 'L2':
        criterion = F.mse_loss
    else:
        raise ValueError(f"Invalid loss type: {config.loss_type}")

    optimizer = optim.Adam(model.parameters())

    print(f'Model param count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    epoch_start = 0
    train_losses = []
    val_losses = []
    val_equiv_errors = []
    val_rel_equiv_errors = []
    val_so3_equiv_errors = []
    val_so3_rel_equiv_errors = []
    best_val_loss = float('inf') 
    best_model = None
    best_epoch = 0
    val_loss = 0.0
    equiv_err = 0
    rel_equiv_err = 0
    so3_equiv_err = 0
    so3_rel_equiv_err = 0

    train_input_files, train_target_files, val_input_files, val_target_files, test_input_files, test_target_files = get_tvt_file_lists(config.dataset)

    train_dataset = NumpyTimeSeriesDataset(
        input_file_list=train_input_files,
        target_file_list=train_target_files,
        means_stds=None,
    )

    val_dataset = NumpyTimeSeriesDataset(
        input_file_list=val_input_files,
        target_file_list=val_target_files,
        means_stds=train_dataset.get_means_stds(),
        augmentation_group=config.val_aug_group
    )

    test_dataset = NumpyTimeSeriesDataset(
        input_file_list=test_input_files,
        target_file_list=test_target_files,
        means_stds=train_dataset.get_means_stds(),
        augmentation_group=config.test_aug_group
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
            batch_size=1, 
            num_workers=1,
            pin_memory=True
        )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        num_workers=1,
        pin_memory=True
    )

    for epoch in range(config.epochs):
        learning_rate = config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate          
            
        model.train()
            
        running_loss = 0.0
            
        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            if config.train_aug_group != None:
                if config.train_aug_group == "oct":
                    inputs, targets = apply_octahedral_augmentation(inputs, targets)
                elif config.train_aug_group == "so3":
                    inputs, targets = apply_so3_augmentation(inputs, targets)
                else:
                    raise ValueError(f"Invalid augmentation group: {config.augmentation_group}")

            pred = model(inputs)
            loss = criterion(pred, targets)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = running_loss / len(train_loader)

        if epoch % config['validation_interval'] == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                equiv_err = 0
                rel_equiv_err = 0
                so3_equiv_err = 0
                so3_rel_equiv_err = 0
                for inputs, targets in tqdm(val_loader, desc="Validation loss"):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    pred = model(inputs)
                    # error = octahedral_equivariance_error(inputs, pred, targets, model, batch_size=config.eval_batch_size)
                    val_loss += criterion(pred, targets).item()

                    if config.track_oct_equivariance_error:
                        curr_equiv_err, curr_rel_equiv_err = mean_octahedral_equivariance_error(inputs, pred, targets, model, batch_size=config.eval_batch_size)
                        equiv_err += curr_equiv_err.item()  
                        rel_equiv_err += curr_rel_equiv_err.item()  
                    
                    if config.track_so3_equivariance_error:
                        curr_so3_equiv_err, curr_so3_rel_equiv_err = mean_so3_equivariance_error(inputs, pred, targets, model, batch_size=config.eval_batch_size, rotation_batch_size=config.rotation_batch_size)
                        so3_equiv_err += curr_so3_equiv_err.item()
                        so3_rel_equiv_err += curr_so3_rel_equiv_err.item()
                    
                # average over validation batches (each call already averages over rotations)
                equiv_err = equiv_err / len(val_loader)
                rel_equiv_err = rel_equiv_err / len(val_loader)
                so3_equiv_err = so3_equiv_err / len(val_loader)
                so3_rel_equiv_err = so3_rel_equiv_err / len(val_loader)
                val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_equiv_errors.append(equiv_err)
        val_rel_equiv_errors.append(rel_equiv_err)
        val_so3_equiv_errors.append(so3_equiv_err)
        val_so3_rel_equiv_errors.append(so3_rel_equiv_err)

        label = "Octahedral" if val_dataset[0][0].dim() == 4 else "C4"
        print(f"Epoch [{epoch:4d}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        if config.track_oct_equivariance_error:
            print(f"  Oct Equiv Error: {equiv_err:.4f}, Oct Rel Equiv Error: {rel_equiv_err:.4f}")
        if config.track_so3_equivariance_error:
            print(f"  SO(3) Equiv Error: {so3_equiv_err:.4f}, SO(3) Rel Equiv Error: {so3_rel_equiv_err:.4f}")
    
        if val_loss < best_val_loss:
            save_new_model = True
            # best_model = copy.deepcopy(model)
            best_val_loss = val_loss
            best_epoch = epoch
            # best_model_state_dict = model.state_dict()
            best_state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}


        if epoch % config['save_interval'] == 0:
            plot_losses(train_losses, val_losses, best_epoch, os.path.join(global_config.experiment_outputs, config['barcode'], f'{config.barcode}_loss_plot.png'))
            
            if config.track_oct_equivariance_error or config.track_so3_equivariance_error:    
                plot_equivariance_errors(
                    val_equiv_errors, val_rel_equiv_errors, 
                    val_so3_equiv_errors, val_so3_rel_equiv_errors,
                    os.path.join(global_config.experiment_outputs, config['barcode'], f'{config.barcode}_equiv_errors.png')
                )
                
            if save_new_model and best_state_cpu is not None:
                print(f"----> Saving a new model, from epoch: {best_epoch}, which had val loss: {best_val_loss}. {juicy_word()}!")
            
                torch.save(
                {
                    "model": best_state_cpu,
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "val_equiv_errors": val_equiv_errors,
                    "val_rel_equiv_errors": val_rel_equiv_errors,
                    "val_so3_equiv_errors": val_so3_equiv_errors,
                    "val_so3_rel_equiv_errors": val_so3_rel_equiv_errors,
                    "seed": seed
                },
                os.path.join(global_config.experiment_outputs, config['barcode'], f'{config.barcode}_checkpoint.pt'),
                )
                print(f"----> Running eval on model from Epoch {best_epoch}")

                best_model = get_model(config.model).to(device)
                best_model.load_state_dict({k: v.to(device) for k, v in best_state_cpu.items()})
                best_model.eval()
                with torch.inference_mode():
                    #plot_spectra(model = best_model,
                    #                                 loader = train_loader,
                    #                                 output_path = os.path.join(global_config.experiment_outputs, config['barcode'], f'{config.barcode}_train_equiv_error_spectrum.png'),
                    #                                 title = 'Train')
                    plot_spectra(model = best_model,
                                            loader = val_loader,
                                            output_path = os.path.join(global_config.experiment_outputs, config['barcode'], f'{config.barcode}_val_equiv_error_spectrum.png'),
                                            title = 'Val')
                    visualize_predictions_3D(model = best_model,
                                            loader = train_loader,
                                            output_path = os.path.join(global_config.experiment_outputs, config['barcode'], f'{config.barcode}_train_predictions_bestmodel.png'),
                                            title = 'Train')

                    visualize_predictions_3D(model = best_model,
                                            loader = val_loader,
                                            output_path = os.path.join(global_config.experiment_outputs, config['barcode'], f'{config.barcode}_val_predictions_bestmodel.png'),
                                            title = 'Val')
                    
                    # visualize_feature_refinement(model = best_model,
                    #                         loader = val_loader,
                    #                         output_path = os.path.join(global_config.experiment_outputs, config['barcode'], f'{config.barcode}_val_feature_refinement.png'),
                    #                         title = 'Val Feature Refinement')

                    test_loss = 0.0    
                    for inputs, targets in tqdm(test_loader, desc="Testing"):
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        pred = best_model(inputs)
                        test_loss += criterion(pred, targets).item()
                    test_loss = test_loss / len(test_loader)
                    print(f"Test Loss: {test_loss:.4f}")

                    # Equivariance error
                    equiv_err, rel_equiv_err = mean_octahedral_equivariance_error(inputs, pred, targets, best_model, batch_size=config.eval_batch_size)
                    so3_equiv_err, so3_rel_equiv_err = mean_so3_equivariance_error(inputs, pred, targets, best_model, batch_size=config.eval_batch_size, rotation_batch_size=config.rotation_batch_size)
                    
                    print(f"Test Octahedral Equivariance Error: {equiv_err:.4f}, Test Octahedral Relative Equivariance Error: {rel_equiv_err:.4f}")
                    print(f"Test SO(3) Equivariance Error: {so3_equiv_err:.4f}, Test SO(3) Relative Equivariance Error: {so3_rel_equiv_err:.4f}")

                    # Save test results as dictionary
                    test_results = {        
                        "test_loss": test_loss,
                        "equiv_err": equiv_err,
                        "rel_equiv_err": rel_equiv_err,
                        "so3_equiv_err": so3_equiv_err,
                        "so3_rel_equiv_err": so3_rel_equiv_err
                    }
                    np.save(os.path.join(global_config.experiment_outputs, config['barcode'], f'{config.barcode}_test_results.npy'), test_results)

                del best_model
                torch.cuda.empty_cache()
                save_new_model = False

        # Optional: Separate plots for absolute and relative equivariance errors
        # plot_equivariance_error(val_equiv_errors, os.path.join(global_config.experiment_outputs, config['barcode'], f'{config.barcode}_equiv_errors.png'))
        # plot_equivariance_error(val_rel_equiv_errors, os.path.join(global_config.experiment_outputs, config['barcode'], f'{config.barcode}_rel_equiv_errors.png'))

        # loss_data_path = os.path.join(global_config.experiment_outputs, config['barcode'], f'{config.barcode}_losses.npy')
        # equiv_data_path = os.path.join(global_config.experiment_outputs, config['barcode'], f'{config.barcode}_equiv_errors.npy')
        # loss_data = np.array([train_losses, val_losses]).T
        # equiv_data = np.array([val_equiv_errors, val_rel_equiv_errors]).T
        # np.save(loss_data_path, loss_data)
        # np.save(equiv_data_path, equiv_data)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # required on macOS/Python 3.13
    main()