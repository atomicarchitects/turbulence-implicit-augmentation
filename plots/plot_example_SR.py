from re import T
import torch
import numpy as np
import os
import torch.optim as optim
torch.set_default_dtype(torch.float32)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle

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
    plot_spectrum,
    visualize_predictions_3D,
    mean_octahedral_equivariance_error,
    plot_equivariance,
    eval_equivariance
)
import argparse
import global_config as global_config
from transformations import apply_octahedral_augmentation, apply_so3_augmentation, rotate_octahedral_exact, get_all_octahedral_angles
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

def get_example_predictions(experiment_dir, device):
    experiment_dir = experiment_dir
    config = _load_config_label(experiment_dir)
    checkpoint = torch.load(os.path.join(experiment_dir, f'{config.barcode}_checkpoint.pt'), map_location=device)
    best_state = checkpoint["model"]
    best_model = get_model(config.model).to(device)
    best_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    best_model.eval()

    train_input_files, train_target_files, _, _, test_input_files, test_target_files = get_tvt_file_lists(config.dataset)
    train_dataset = NumpyTimeSeriesDataset(
        input_file_list=train_input_files,
        target_file_list=train_target_files,
        scaling_factors=None,
        augmentation_group=config.train_aug_group,
        scalar_predictor=config.scalar_predictor,
    )
    test_dataset = NumpyTimeSeriesDataset(
        input_file_list=test_input_files,
        target_file_list=test_target_files,
        scaling_factors=train_dataset.get_scaling_factors(),
        augmentation_group=config.test_aug_group,
        scalar_predictor=config.scalar_predictor,
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.eval_batch_size, 
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )
    inputs, targets = next(iter(test_loader))
    input_sample  = inputs[0:1].to(device)
    target_sample = targets[0].to(device)
    prediction    = best_model(input_sample)[0]

    # unnormalize
    scaling_factors = train_dataset.get_scaling_factors()
    input_sample = input_sample*scaling_factors['input_std'] + scaling_factors['input_mean']
    target_sample = target_sample*scaling_factors['target_std'] + scaling_factors['target_mean']
    prediction = prediction*scaling_factors['target_std'] + scaling_factors['target_mean']

    return input_sample[0].detach().cpu().numpy(), target_sample[0].detach().cpu().numpy(), prediction[0].detach().cpu().numpy()

def get_example_fields(experiment_dir, device):
    experiment_dir = experiment_dir
    config = _load_config_label(experiment_dir)
    checkpoint = torch.load(os.path.join(experiment_dir, f'{config.barcode}_checkpoint.pt'), map_location=device)
    best_state = checkpoint["model"]
    best_model = get_model(config.model).to(device)
    best_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    best_model.eval()

    train_input_files, train_target_files, _, _, test_input_files, test_target_files = get_tvt_file_lists(config.dataset)
    train_dataset = NumpyTimeSeriesDataset(
        input_file_list=train_input_files,
        target_file_list=train_target_files,
        scaling_factors=None,
        augmentation_group=config.train_aug_group,
        scalar_predictor=config.scalar_predictor,
    )
    test_dataset = NumpyTimeSeriesDataset(
        input_file_list=test_input_files,
        target_file_list=test_target_files,
        scaling_factors=train_dataset.get_scaling_factors(),
        augmentation_group=config.test_aug_group,
        scalar_predictor=config.scalar_predictor,
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.eval_batch_size, 
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )
    input_sample1 = test_dataset[0][0].unsqueeze(0).to(device)  # unsqueeze to add batch dim
    target_sample1 = test_dataset[0][1].to(device)
    input_sample2 = test_dataset[10][0].unsqueeze(0).to(device)
    target_sample2 = test_dataset[10][1].to(device)

    input_sample3 = test_dataset[20][0].unsqueeze(0).to(device)
    target_sample3 = test_dataset[20][1].to(device)

    input_sample4 = test_dataset[100][0].unsqueeze(0).to(device)
    target_sample4 = test_dataset[100][1].to(device)


    input_files, target_files = get_test_file_lists('channel5200_nearwall_filtered_fs4', 'channel5200_nearwall')
    dataset = NumpyTimeSeriesDataset(
        input_file_list=input_files,
        target_file_list=target_files,
        scaling_factors=train_dataset.get_scaling_factors(),
        augmentation_group=config.test_aug_group,
        scalar_predictor=config.scalar_predictor,
    )

    input_sample5 = dataset[0][0].unsqueeze(0).to(device)
    target_sample5 = dataset[0][1].to(device)


    # unnormalize
    scaling_factors = train_dataset.get_scaling_factors()

    input_sample1 = input_sample1*scaling_factors['input_std'] + scaling_factors['input_mean']
    target_sample1 = target_sample1*scaling_factors['target_std'] + scaling_factors['target_mean']
    input_sample2 = input_sample2*scaling_factors['input_std'] + scaling_factors['input_mean']
    target_sample2 = target_sample2*scaling_factors['target_std'] + scaling_factors['target_mean']
    input_sample3 = input_sample3*scaling_factors['input_std'] + scaling_factors['input_mean']
    target_sample3 = target_sample3*scaling_factors['target_std'] + scaling_factors['target_mean']
    input_sample4 = input_sample4*scaling_factors['input_std'] + scaling_factors['input_mean']
    target_sample4 = target_sample4*scaling_factors['target_std'] + scaling_factors['target_mean']
    input_sample5 = input_sample5*scaling_factors['input_std'] + scaling_factors['input_mean']
    target_sample5 = target_sample5*scaling_factors['target_std'] + scaling_factors['target_mean']
    return target_sample1[0].detach().cpu().numpy(), target_sample2[0].detach().cpu().numpy(), target_sample3[0].detach().cpu().numpy(), target_sample4[0].detach().cpu().numpy(), target_sample5[0].detach().cpu().numpy()


def get_example_equivariance(x, experiment_dir, device, angle, means_stds):
    experiment_dir = experiment_dir
    config = _load_config_label(experiment_dir)
    checkpoint = torch.load(os.path.join(experiment_dir, f'{config.barcode}_checkpoint.pt'), map_location=device)
    best_state = checkpoint["model"]
    best_model = get_model(config.model).to(device)
    best_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    best_model.eval()

    fx    = best_model(x)[0]
    gx = rotate_octahedral_exact(x, angle, rotate_channels=True, mode="pairwise")
    fgx = best_model(gx)[0]
    gfx = rotate_octahedral_exact(fx, angle, rotate_channels=True, mode="pairwise")

    x = x*means_stds['input_std'] + means_stds['input_mean']
    gx = gx*means_stds['input_std'] + means_stds['input_mean']
    fx = fx*means_stds['target_std'] + means_stds['target_mean']
    fgx = fgx*means_stds['target_std'] + means_stds['target_mean']
    gfx = gfx*means_stds['target_std'] + means_stds['target_mean']


    return x[0].detach().cpu().numpy(), gx[0].detach().cpu().numpy(), fx[0].detach().cpu().numpy(), fgx[0].detach().cpu().numpy(), gfx[0].detach().cpu().numpy()

def get_example_x(experiment_dir, device):
    experiment_dir = experiment_dir
    config = _load_config_label(experiment_dir)

    train_input_files, train_target_files, _, _, test_input_files, test_target_files = get_tvt_file_lists(config.dataset)
    train_dataset = NumpyTimeSeriesDataset(
        input_file_list=train_input_files,
        target_file_list=train_target_files,
        scaling_factors=None,
        augmentation_group=config.train_aug_group,
        scalar_predictor=config.scalar_predictor,
    )
    test_dataset = NumpyTimeSeriesDataset(
        input_file_list=test_input_files,
        target_file_list=test_target_files,
        scaling_factors=train_dataset.get_scaling_factors(),
        augmentation_group=config.test_aug_group,
        scalar_predictor=config.scalar_predictor,
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.eval_batch_size, 
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )
    inputs, _ = next(iter(test_loader))
    x  = inputs[0:1].to(device)
    return x, train_dataset.get_scaling_factors()


def visualize_example_fields(output_dir, output_file, experiment_dir_nw, experiment_dir_md, device):
    target_sample1, target_sample2, target_sample3, target_sample4, target_sample5 = get_example_fields(experiment_dir_nw, device)
    cmap = 'plasma'
    vmin = 0.8
    vmax = 1.2
    slice_fine = (slice(None), slice(None), slice(None), 29)
    example_xy1 = np.sqrt(np.sum(target_sample1[slice_fine]**2, axis=0)).T
    example_xy2 = np.sqrt(np.sum(target_sample2[slice_fine]**2, axis=0)).T
    example_xy3 = np.sqrt(np.sum(target_sample3[slice_fine]**2, axis=0)).T
    example_xy4 = np.sqrt(np.sum(target_sample4[slice_fine]**2, axis=0)).T
    example_xy5 = np.sqrt(np.sum(target_sample5[slice_fine]**2, axis=0)).T
    fig,ax = plt.subplots(1,5,figsize=(4,1.5))

    ax[0].imshow(example_xy1, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[1].imshow(example_xy2, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[2].imshow(example_xy3, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[3].imshow(example_xy4, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[4].imshow(example_xy5, cmap=cmap, vmin=vmin, vmax=vmax)

    for ax in ax:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_file), dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.close()



def visualize_example_predictions(output_dir, output_file, experiment_dir_nw, experiment_dir_md, device,  cmap_magnitude, vmin_magnitude, vmax_magnitude, cmap_diff):
    input_nw, target_nw, prediction_nw = get_example_predictions(experiment_dir_nw, device)
    input_md, target_md, prediction_md = get_example_predictions(experiment_dir_md, device)

    vmin_diff = 0
    vmax_diff = 0.03
    slice_fine = (slice(None), slice(None), slice(None), 29)
    slice_coarse = (slice(None), slice(None), slice(None), 7)

    # xy slices.
    inp_xy_nw = np.sqrt(np.sum(input_nw[slice_coarse]**2, axis=0)).T      
    tgt_xy_nw = np.sqrt(np.sum(target_nw[slice_fine]**2, axis=0)).T      
    pred_xy_nw = np.sqrt(np.sum(prediction_nw[slice_fine]**2, axis=0)).T    

    inp_xy_md = np.sqrt(np.sum(input_md[slice_coarse]**2, axis=0)).T      
    tgt_xy_md = np.sqrt(np.sum(target_md[slice_fine]**2, axis=0)).T      
    pred_xy_md = np.sqrt(np.sum(prediction_md[slice_fine]**2, axis=0)).T    

    # Absolute difference plots
    diff_xy_nw = np.abs(pred_xy_nw - tgt_xy_nw)
    diff_xy_md = np.abs(pred_xy_md - tgt_xy_md)

    # Plot example magnitude slices (3 plots + difference)
    fig = plt.figure(figsize=(5.2, 1.5))
    gs = gridspec.GridSpec(3, 7, width_ratios=[1, 1, 1, 1, 0.07, 1, 0.07], height_ratios=[0.12, 1, 1], wspace=0.3, hspace=0.15)

    ax_png = fig.add_subplot(gs[1:, 0])
    img = plt.imread('outputs/plots_final/legend.png')
    ax_png.imshow(img)
    ax_png.axis('off')

    # Column headers
    ax_header1 = fig.add_subplot(gs[0, 1])
    ax_header1.text(0.5, 0.5, 'Input', ha='center', va='center', fontsize=9)
    ax_header1.axis('off')
    
    ax_header2 = fig.add_subplot(gs[0, 2])
    ax_header2.text(0.5, 0.5, 'Predicted', ha='center', va='center', fontsize=9)
    ax_header2.axis('off')
    
    ax_header3 = fig.add_subplot(gs[0, 3])
    ax_header3.text(0.5, 0.5, 'Target', ha='center', va='center', fontsize=9)
    ax_header3.axis('off')

    ax_header4 = fig.add_subplot(gs[0, 5])
    ax_header4.text(0.5, 0.5, 'Abs. Error', ha='center', va='center', fontsize=9)
    ax_header4.axis('off')

    ax0 = fig.add_subplot(gs[1, 1])
    ax1 = fig.add_subplot(gs[1, 2])
    ax2 = fig.add_subplot(gs[1, 3])
    ax6 = fig.add_subplot(gs[1, 5])  # Difference near-wall

    ax3 = fig.add_subplot(gs[2, 1])
    ax4 = fig.add_subplot(gs[2, 2])
    ax5 = fig.add_subplot(gs[2, 3])
    ax7 = fig.add_subplot(gs[2, 5])  # Difference middle

    # Row labels
    ax0.set_ylabel('Near-wall', fontsize=9, color='blue')
    ax3.set_ylabel('Middle', fontsize=9, color='red')

    # Colorbar for magnitude (spanning both rows)
    cax_mag = fig.add_subplot(gs[1:, 4])

    # Colorbar for difference (spanning both rows)
    cax_diff = fig.add_subplot(gs[1:, 6])

    # Magnitude plots
    im0 = ax0.imshow(inp_xy_nw, vmin=vmin_magnitude, vmax=vmax_magnitude, cmap=cmap_magnitude, origin='lower')
    im1 = ax1.imshow(pred_xy_nw, vmin=vmin_magnitude, vmax=vmax_magnitude, cmap=cmap_magnitude, origin='lower')
    im2 = ax2.imshow(tgt_xy_nw, vmin=vmin_magnitude, vmax=vmax_magnitude, cmap=cmap_magnitude, origin='lower')
    im3 = ax3.imshow(inp_xy_md, vmin=vmin_magnitude, vmax=vmax_magnitude, cmap=cmap_magnitude, origin='lower')
    im4 = ax4.imshow(pred_xy_md, vmin=vmin_magnitude, vmax=vmax_magnitude, cmap=cmap_magnitude, origin='lower')
    im5 = ax5.imshow(tgt_xy_md, vmin=vmin_magnitude, vmax=vmax_magnitude, cmap=cmap_magnitude, origin='lower')

    # Absolute difference plots
    im6 = ax6.imshow(diff_xy_nw, vmin=vmin_diff, vmax=vmax_diff, cmap=cmap_diff, origin='lower')
    im7 = ax7.imshow(diff_xy_md, vmin=vmin_diff, vmax=vmax_diff, cmap=cmap_diff, origin='lower')

    for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    for ax in [ax0, ax1, ax2, ax6]:
        for spine in ax.spines.values():
            spine.set_color('blue')
            spine.set_linewidth(1)

    for ax in [ax3, ax4, ax5, ax7]:
        for spine in ax.spines.values():
            spine.set_color('red')
            spine.set_linewidth(1)

    # Magnitude colorbar
    norm_mag = mcolors.Normalize(vmin=vmin_magnitude, vmax=vmax_magnitude)
    sm_mag = ScalarMappable(norm=norm_mag, cmap=cmap_magnitude)
    cbar_mag = fig.colorbar(sm_mag, cax=cax_mag)
    cbar_mag.locator = ticker.MultipleLocator(0.2)
    cbar_mag.update_ticks()
    cbar_mag.set_label(r'$| \mathbf{U} |$ (m/s)', labelpad=15, fontsize=9)

    # Difference colorbar
    norm_diff = mcolors.Normalize(vmin=vmin_diff, vmax=vmax_diff)
    sm_diff = ScalarMappable(norm=norm_diff, cmap=cmap_diff)
    cbar_diff = fig.colorbar(sm_diff, cax=cax_diff)
    cbar_diff.locator = ticker.MultipleLocator(vmax_diff/2)  # Adjust tick spacing as needed
    cbar_diff.update_ticks()
    cbar_diff.set_label(r'$| \Delta \mathbf{U} |$ (m/s)', labelpad=15, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_file), dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.close()
            
def visualize_equivariance(output_dir, output_file, experiment_dir_noaug, experiment_dir_aug, device,  cmap_vz, vmin_uz, vmax_uz, cmap_diff,vmin_diff, vmax_diff, cborder, angle_equivariance, label_fontsize):
    x_torch, means_stds = get_example_x(experiment_dir_noaug, device)

    angle_equivariance = torch.tensor([[np.pi/2, 3*np.pi/2, 3*np.pi/2]])
    x, gx, fx_noaug, fgx_noaug, gfx_noaug = get_example_equivariance(x_torch, experiment_dir_noaug, device, angle=angle_equivariance, means_stds=means_stds)
    _, _, fx_aug, fgx_aug, gfx_aug = get_example_equivariance(x_torch, experiment_dir_aug, device, angle=angle_equivariance, means_stds=means_stds)


    slice_fine = (2, slice(None), slice(None), 29)
    slice_coarse = (2, slice(None), slice(None), 7)

    x_xy = x[slice_coarse].T#np.sqrt(np.sum(x[:, 7, :, :]**2, axis=0))     
    gx_xy = gx[slice_coarse].T#np.sqrt(np.sum(gx[:, 7, :, :]**2, axis=0))  

    fx_xy_noaug = fx_noaug[slice_fine].T#np.sqrt(np.sum(fx_noaug[:, 7, :, :]**2, axis=0))
    fgx_xy_noaug = fgx_noaug[slice_fine].T#np.sqrt(np.sum(fgx_noaug[:, 29, :, :]**2, axis=0)) 
    gfx_xy_noaug = gfx_noaug[slice_fine].T#np.sqrt(np.sum(gfx_noaug[:, 29, :, :]**2, axis=0))

    fgx_xy_aug = fgx_aug[slice_fine].T#np.sqrt(np.sum(fgx_aug[:, 29, :, :]**2, axis=0)) 
    fx_xy_aug = fx_aug[slice_fine].T#np.sqrt(np.sum(fx_aug[:, 7, :, :]**2, axis=0))
    gfx_xy_aug = gfx_aug[slice_fine].T#np.sqrt(np.sum(gfx_aug[:, 29, :, :]**2, axis=0))

    fig = plt.figure(figsize=(5.8, 1.8))
    plt.tight_layout()
    gs = gridspec.GridSpec(2, 10, width_ratios=[1, 1, 1, 1, 1, 0.05, 0.7, 1, 0.6, 0.5], wspace=0.3, hspace=0.3)

    # cbars span both rows


    # individual axes
    ax_x = fig.add_subplot(gs[:, 0])
    ax_x.set_xlabel(r'$\:\,\,\,[\mathbf{x}]_{z}$' + '\n' + r'(a)', fontsize=label_fontsize)
    ax_gx = fig.add_subplot(gs[:, 1])
    ax_gx.set_xlabel(r'$\:\,\,[g\cdot\mathbf{x}]_{z}$' + '\n' + r'(b)', fontsize=label_fontsize)


    ax_r1_fx = fig.add_subplot(gs[0, 2])
    ax_r1_fx.set_xlabel(r'$\:\,\,[\mathbf{f}\,(\mathbf{x})]_z$' + '\n' + r'(c)', fontsize=label_fontsize)
    ax_r2_fx = fig.add_subplot(gs[1, 2])
    ax_r2_fx.set_xlabel(r'$\:\,\,[\mathbf{f}\,(\mathbf{x})]_z$' + '\n' + r'(g)', fontsize=label_fontsize)

    ax_r1_fgx = fig.add_subplot(gs[0, 3])
    ax_r1_fgx.set_xlabel(r'$\:\,\,[\mathbf{f}\,(g\cdot\mathbf{x})]_z$' + '\n' + r'(d)', fontsize=label_fontsize)
    ax_r2_fgx = fig.add_subplot(gs[1, 3])
    ax_r2_fgx.set_xlabel(r'$\:\,\,[\mathbf{f}\,(g\cdot\mathbf{x})]_z$' + '\n' + r'(h)', fontsize=label_fontsize)

    ax_r1_gfx = fig.add_subplot(gs[0, 4])
    ax_r1_gfx.set_xlabel(r'$\:\,\,[g\cdot \mathbf{f}\,(\mathbf{x})]_z$' + '\n' + r'(e)', fontsize=label_fontsize)
    ax_r2_gfx = fig.add_subplot(gs[1, 4])
    ax_r2_gfx.set_xlabel(r'$\:\,\,[g\cdot \mathbf{f}\,(\mathbf{x})]_z$' + '\n' + r'(i)', fontsize=label_fontsize)

    ax_r1_diff = fig.add_subplot(gs[0, 7])
    ax_r1_diff.set_xlabel(r'$\:\,\,|[\mathbf{f}\,(g\cdot\mathbf{x}) - g\cdot \mathbf{f}\,(\mathbf{x})]_z|$' + '\n' + r'(f)', fontsize=label_fontsize)
    ax_r2_diff = fig.add_subplot(gs[1, 7])
    ax_r2_diff.set_xlabel(r'$\:\,\,|[\mathbf{f}\,(g\cdot\mathbf{x}) - g\cdot \mathbf{f}\,(\mathbf{x})]_z|$' + '\n' + r'(j)', fontsize=label_fontsize)
    
    im0 = ax_x.imshow(x_xy, vmin=vmin_uz, vmax=vmax_uz, cmap=cmap_vz, origin='lower')
    im1 = ax_gx.imshow(gx_xy, vmin=vmin_uz, vmax=vmax_uz, cmap=cmap_vz, origin='lower')


    im2 = ax_r1_fx.imshow(fx_xy_noaug, vmin=vmin_uz, vmax=vmax_uz, cmap=cmap_vz, origin='lower')
    im3 = ax_r1_fgx.imshow(fgx_xy_noaug, vmin=vmin_uz, vmax=vmax_uz, cmap=cmap_vz, origin='lower')
    im4 = ax_r1_gfx.imshow(gfx_xy_noaug, vmin=vmin_uz, vmax=vmax_uz, cmap=cmap_vz, origin='lower')
    im5 = ax_r2_fx.imshow(fx_xy_aug, vmin=vmin_uz, vmax=vmax_uz, cmap=cmap_vz, origin='lower')
    im6 = ax_r2_fgx.imshow(fgx_xy_aug, vmin=vmin_uz, vmax=vmax_uz, cmap=cmap_vz, origin='lower')   
    im7 = ax_r2_gfx.imshow(gfx_xy_aug, vmin=vmin_uz, vmax=vmax_uz, cmap=cmap_vz, origin='lower')

    im8 = ax_r1_diff.imshow(np.abs(fgx_xy_noaug - gfx_xy_noaug), vmin=vmin_diff, vmax=vmax_diff, cmap=cmap_diff, origin='lower')
    im9 = ax_r2_diff.imshow(np.abs(fgx_xy_aug - gfx_xy_aug), vmin=vmin_diff, vmax=vmax_diff, cmap=cmap_diff, origin='lower')

    for ax in [ax_x, ax_gx, ax_r1_fx, ax_r1_fgx, ax_r1_gfx, ax_r2_fx, ax_r2_fgx, ax_r2_gfx, ax_r1_diff, ax_r2_diff]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    for ax in [ax_x, ax_gx, ax_r1_fx, ax_r1_fgx, ax_r1_gfx, ax_r2_fx, ax_r2_fgx, ax_r2_gfx, ax_r1_diff, ax_r2_diff]:
        for spine in ax.spines.values():
            spine.set_color(cborder)
            spine.set_linewidth(1)

    for ax in [ax_x, ax_r1_fx, ax_r2_fx]: # axes with wall on top
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Add hatching to the left side (vertical wall)
        hatch = Rectangle((xlim[0], ylim[1]), 
                            width=xlim[1]-xlim[0],
                            height=(ylim[1]-ylim[0])*0.05,  # 5% of height
                            transform=ax.transData,
                            fill=False,
                            hatch='/////////',  # angled lines (use '|||' for vertical, '---' for horizontal)
                            edgecolor=cborder,
                            linewidth=0,
                            clip_on=False)
        ax.add_patch(hatch)

    for ax in [ax_gx, ax_r1_fgx, ax_r1_gfx, ax_r2_fgx, ax_r2_gfx, ax_r1_diff, ax_r2_diff]: # axes with wall on left side
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Add hatching to the left of the left side (outside the plot area)
        hatch = Rectangle((xlim[0] - (xlim[1]-xlim[0])*0.05, ylim[0]), 
                            width=(xlim[1]-xlim[0])*0.05,  # 5% of width
                            height=ylim[1]-ylim[0],
                            transform=ax.transData,
                            fill=False,
                            hatch='/////////',  # more slashes = tighter spacing
                            edgecolor=cborder,
                            linewidth=0,
                            clip_on=False)  # Allow drawing outside axis
        ax.add_patch(hatch)
    pos_top = ax_r1_gfx.get_position()
    pos_bottom = ax_r2_gfx.get_position()

    # Create first colorbar axes [left, bottom, width, height]
    cax1 = fig.add_axes([pos_top.x1 + 0.02, pos_bottom.y0, 0.01, pos_top.y1 - pos_bottom.y0])

    pos_top2 = ax_r1_diff.get_position()
    pos_bottom2 = ax_r2_diff.get_position()

    # Create second colorbar axes
    cax2 = fig.add_axes([pos_top2.x1 + 0.05, pos_bottom2.y0, 0.01, pos_top2.y1 - pos_bottom2.y0])

    norm = mcolors.Normalize(vmin=vmin_uz, vmax=vmax_uz)
    sm = ScalarMappable(norm=norm, cmap=cmap_vz)
    cbar1 = fig.colorbar(sm, cax=cax1)
    cbar1.locator = ticker.MultipleLocator(vmax_uz)  # tick every 1.0
    cbar1.update_ticks()
    cbar1.ax.set_xlabel(r'$\:\,\,[\mathbf{U}]_z $' + '\n' + r'(m/s)', fontsize=label_fontsize)
    cbar1.ax.tick_params(labelsize=label_fontsize)

    norm = mcolors.Normalize(vmin=vmin_diff, vmax=vmax_diff)
    sm = ScalarMappable(norm=norm, cmap=cmap_diff)
    cbar2 = fig.colorbar(sm, cax=cax2)
    cbar2.locator = ticker.MultipleLocator(vmax_diff/2)  # tick every 1.0
    cbar2.update_ticks()
    cbar2.ax.set_xlabel(r'diff.' + '\n' + r'(m/s)', fontsize=label_fontsize)
    cbar2.ax.tick_params(labelsize=label_fontsize)


    # Add arrow annotation for top row (No augmentation)
    fig.text(0.315, 0.81, r'No aug. $\rightarrow$', 
            fontsize=label_fontsize, 
            va='center', 
            ha='right')

    # Add arrow annotation for bottom row (With augmentation)
    fig.text(0.315, 0.18, r'Aug. $\rightarrow$', 
            fontsize=label_fontsize, 
            va='center', 
            ha='right')



    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_file), dpi=1000, bbox_inches='tight',pad_inches=0)
    plt.close()
            