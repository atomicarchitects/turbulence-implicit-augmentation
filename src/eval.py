import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from transformations import unstandardize, rand_so3_angles, get_all_octahedral_angles, rotate_3d, rotate_octahedral_exact, euler_angles_to_matrix


def _reshape_rotations(tensor: torch.Tensor, num_rots: int, batch: int) -> torch.Tensor:
    return tensor.reshape(num_rots, batch, *tensor.shape[1:])


def _relative_equivariance_error(diff_norm: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    # [24, B, C, D, H, W]
    reference_norm = torch.linalg.norm(reference, dim=1, keepdim=True) # [24, B, C, D, H, W]
    denom = reference_norm.clamp_min(1e-8)
    return diff_norm / denom


def eval_test_set(config, loader, model, criterion, scaling_factors):
    device = next(model.parameters()).device
    test_loss = 0.0

    for inputs, targets in tqdm.tqdm(loader, desc="Plain eval test set"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        pred = model(inputs)

        # Get the loss in physical units
        test_loss += criterion(unstandardize(pred, mean = scaling_factors['target_mean'], std = scaling_factors['target_std']),
                               unstandardize(targets, mean = scaling_factors['target_mean'], std = scaling_factors['target_std'])).item()

    test_loss = test_loss / len(loader)
    return test_loss 

def eval_equivariance(config, loader, model, compute_spectra, scaling_factors):
    device = next(model.parameters()).device
    abs_equiv_error = 0.0
    abs_equiv_error_u = 0.0
    abs_equiv_error_v = 0.0
    abs_equiv_error_w = 0.0
    spec_error = []
    for inputs, targets in tqdm.tqdm(loader, desc="Equivariance eval test set"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        pred = model(inputs)
        # [len(inputs), 24, C, D, H, W] (angle wise, channelwise, pointwise absolute error)
        equiv_error = octahedral_equivariance_error(inputs, pred, targets, model, batch_size=config.rotation_batch_size)

        # Convert equivariance error to physical units
        equiv_error = unstandardize(equiv_error,
                                    mean = 0, # equiv_error is a difference
                                    std = scaling_factors['target_std'].unsqueeze(1)) # unsqueeze for correct broadcasting
        
        # Get spectra of equiv error
        if compute_spectra:
            k, spec_i = compute_isotropic_spectrum(equiv_error.flatten(0,1), Lx=1)
            spec_error.append(spec_i)

        # Reduce to pointwise norm error [len(inputs), 24, D, H, W]
        equiv_error_pointwise = torch.linalg.norm(equiv_error, dim=2)
        equiv_error_u = abs(equiv_error[:,:,0,:,:,:])
        equiv_error_v = abs(equiv_error[:,:,1,:,:,:])
        equiv_error_w = abs(equiv_error[:,:,2,:,:,:])
        abs_equiv_error += torch.mean(equiv_error_pointwise).item()
        abs_equiv_error_u += torch.mean(equiv_error_u).item()
        abs_equiv_error_v += torch.mean(equiv_error_v).item()
        abs_equiv_error_w += torch.mean(equiv_error_w).item()

    abs_equiv_error = abs_equiv_error / len(loader)
    abs_equiv_error_u = abs_equiv_error_u / len(loader)
    abs_equiv_error_v = abs_equiv_error_v / len(loader)
    abs_equiv_error_w = abs_equiv_error_w / len(loader)
    if compute_spectra:
        spec_error = np.mean(spec_error, axis=0)
        return abs_equiv_error, (k, spec_error), abs_equiv_error_u, abs_equiv_error_v, abs_equiv_error_w
    else:
        return abs_equiv_error, abs_equiv_error_u, abs_equiv_error_v, abs_equiv_error_w

def octahedral_equivariance_error(inputs, pred, targets, model, batch_size=None):
    """
    Compute f(g·x) - g·f(x) for the 24 rotations in the octahedral group.

    Returns tensors shaped ``[len(inputs), 24, C, D, H, W]`` (angle wise, channelwise, pointwise absolute error).
    """
    num_samples = inputs.shape[0]
    batch_size = num_samples if batch_size is None else min(num_samples, batch_size)
    
    all_angles = get_all_octahedral_angles().to(inputs.device)  # [24, 3]
    num_rots = all_angles.shape[0]

    abs_errors = []
    rel_errors = []

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        inputs_batch = inputs[start:end]
        pred_batch = pred[start:end]
        batch_len = inputs_batch.shape[0]

        rotated_inputs = rotate_octahedral_exact(
            inputs_batch,
            all_angles,
            rotate_channels=True,
            mode="cartesian",
        )
        rotated_preds = rotate_octahedral_exact(
            pred_batch,
            all_angles,
            rotate_channels=True,
            mode="cartesian",
        )

        rotated_then_pred = model(rotated_inputs)
        diff = torch.abs(rotated_then_pred - rotated_preds)
        abs_errors.append(diff)

    error = torch.stack(abs_errors, dim=0)
    return error


def mean_octahedral_equivariance_error(inputs, pred, targets, model, batch_size=None):
    absolute_error, relative_error = octahedral_equivariance_error(
        inputs, pred, targets, model, batch_size=batch_size
    )
    return torch.mean(absolute_error), torch.mean(relative_error)


def so3_equivariance_error(
    inputs,
    pred,
    targets,
    model,
    batch_size=None,
    num_rotations: int = 100,
    rotation_batch_size: int = 32,
):
    """
    Evaluate |f(g·x) - g·f(x)| for random SO(3) rotations.

    Returns tensors shaped ``[num_rotations, B, C, D, H, W]`` (absolute and relative errors).
    """
    num_samples = inputs.shape[0]
    batch_size = num_samples if batch_size is None else min(num_samples, batch_size)
    rotation_batch_size = num_rotations if rotation_batch_size is None else min(num_rotations, rotation_batch_size)
    random_angles = rand_so3_angles(num_rotations).to(inputs.device)  # [num_rotations, 3]

    abs_errors = []
    rel_errors = []

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        inputs_batch = inputs[start:end]
        pred_batch = pred[start:end]
        batch_len = inputs_batch.shape[0]

        batch_abs_errors = []
        batch_rel_errors = []

        for rot_start in range(0, num_rotations, rotation_batch_size):
            rot_end = min(rot_start + rotation_batch_size, num_rotations)
            angles_chunk = random_angles[rot_start:rot_end]
            chunk_rots = angles_chunk.shape[0]

            rotated_inputs = rotate_3d(inputs_batch, angles=angles_chunk)  # [chunk_rots * batch, ...]
            rotated_preds = rotate_3d(pred_batch, angles=angles_chunk)

            rotated_then_pred = model(rotated_inputs)

            rotated_then_pred = _reshape_rotations(rotated_then_pred, chunk_rots, batch_len)
            rotated_preds = _reshape_rotations(rotated_preds, chunk_rots, batch_len)

            diff = rotated_then_pred - rotated_preds
            batch_abs_errors.append(diff.abs())
            batch_rel_errors.append(_relative_equivariance_error(diff, rotated_preds))

        abs_errors.append(torch.cat(batch_abs_errors, dim=0))
        rel_errors.append(torch.cat(batch_rel_errors, dim=0))

    error = torch.cat(abs_errors, dim=1)
    rel_error = torch.cat(rel_errors, dim=1)
    return error, rel_error


def mean_so3_equivariance_error(inputs, pred, targets, model, batch_size=None, num_rotations=100, rotation_batch_size=32):
    """
    Compute the mean absolute SO(3) equivariance error.
    
    Args:
        inputs: [B, C_in, D, H, W]
        pred: [B, C_out, D', H', W'] 
        targets: [B, C_out, D', H', W'] (not used but kept for API consistency)
        model: The model to evaluate
        relative: If True, compute relative error
        batch_size: Batch size for memory management
        num_rotations: Number of random SO(3) rotations to test
        rotation_batch_size: Batch size for rotation processing to save memory
        
    Returns:
        Scalar tensor with mean equivariance error
    """
    absolute_error, relative_error = so3_equivariance_error(
        inputs,
        pred,
        targets,
        model,
        batch_size=batch_size,
        num_rotations=num_rotations,
        rotation_batch_size=rotation_batch_size
    )
    return torch.mean(absolute_error), torch.mean(relative_error)


def visualize_predictions(model, val_loader, output_path):
    model.eval()
    torch.manual_seed(42)  # Same 3 examples every time
    
    with torch.no_grad():
        # Get first batch
        inputs, targets = next(iter(val_loader))
        
        # Take first 3 samples and move to device
        inputs = inputs[:3]
        targets = targets[:3]
        predictions = model(inputs)
        
        # Plot
        fig, axes = plt.subplots(3, 6, figsize=(15, 8))
        
        for i in range(3):
            # Convert to numpy and handle channels
            inp = inputs[i].cpu().numpy()
            tgt = targets[i].cpu().numpy()
            pred = predictions[i].cpu().numpy()
            #pred_then_rotated = torch.rot90(predictions[i],k=1,dims=(1,2))[0].cpu().numpy()
            pred_then_rotated_oct = rotate_octahedral_exact(predictions[i], 1, rotate_channels=True, mode="cartesian")
            rotated_inputs = rotate_octahedral_exact(inputs[i], 1, rotate_channels=True, mode="cartesian")
            #rotated_then_predicted = model(torch.rot90(inputs[i].unsqueeze(0),k=1,dims=(2,3)))[0][0].cpu().numpy()
            rotated_then_predicted = model(rotated_inputs)[0].cpu().numpy()
            
            # Compute difference for diverging colormap
            diff = rotated_then_predicted - pred_then_rotated_oct
            
            if inp.ndim == 3:  # Remove channel dim if needed
                inp = inp.squeeze() if inp.shape[0] == 1 else inp.transpose(1,2,0)
                tgt = tgt.squeeze() if tgt.shape[0] == 1 else tgt.transpose(1,2,0)
                pred = pred.squeeze() if pred.shape[0] == 1 else pred.transpose(1,2,0)
            
            # Compute shared bounds for all images except difference
            #shared_vmax = max(np.abs(inp).max(), np.abs(tgt).max(), np.abs(pred).max(), 
            #                 np.abs(pred_then_rotated).max(), np.abs(rotated_then_predicted).max())
            #diff_vmax = np.abs(diff).max()
            shared_vmax = 5
            diff_vmax = 1
            # All images except difference use same colorbar bounds
            axes[i,0].imshow(inp, cmap='RdBu_r', vmin=-shared_vmax, vmax=shared_vmax)
            axes[i,1].imshow(tgt, cmap='RdBu_r', vmin=-shared_vmax, vmax=shared_vmax)
            axes[i,2].imshow(pred, cmap='RdBu_r', vmin=-shared_vmax, vmax=shared_vmax)
            axes[i,3].imshow(pred_then_rotated, cmap='RdBu_r', vmin=-shared_vmax, vmax=shared_vmax)
            axes[i,4].imshow(rotated_then_predicted, cmap='RdBu_r', vmin=-shared_vmax, vmax=shared_vmax)
            im = axes[i,5].imshow(diff, cmap='RdBu_r', vmin=-diff_vmax, vmax=diff_vmax)
            
            # Add colorbar for the difference plot
            if i == 0:  # Only add colorbar to first row
                cbar = plt.colorbar(im, ax=axes[i,5], shrink=0.8)
                cbar.set_label('Difference')

            axes[i,0].set_title('Input')
            axes[i,1].set_title('Target')
            axes[i,2].set_title('Prediction')
            axes[i,3].set_title('Pred → Rot 90°')
            axes[i,4].set_title('Rot 90° → Pred')
            axes[i,5].set_title('Difference')
            
            for ax in axes[i]:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

def visualize_equivariance(inputs, pred, targets, model, angle=torch.tensor([0,0,np.pi/2])):
    #angle = get_all_octahedral_angles()[0]
    _, Cin, Din, Hin, Win = inputs.shape
    rotated_input = rotate_octahedral_exact(inputs, angle, rotate_channels=True, mode="cartesian")
    rotated_pred = rotate_octahedral_exact(model(inputs), angle, rotate_channels=True, mode="cartesian")
    rotated_then_pred = model(rotated_input)
    indices = [(0,0,0), (5,6,7), (11,5,12)]

    print(f'Example pairs')
    for spatial_index in indices:
        print(f'Predicted then rotated: {rotated_then_pred[...,*spatial_index].detach().cpu().numpy()}')
        print(f'Rotated then predicted: {rotated_pred[...,*spatial_index].detach().cpu().numpy()}')
        print(f'Difference: {rotated_then_pred[...,*spatial_index].detach().cpu().numpy() - rotated_pred[...,*spatial_index].detach().cpu().numpy()}')   



def plot_equivariance(model, loader, output_path, title, angle=torch.tensor([0,0,np.pi/2])):
    model.eval()
    with torch.no_grad():
        # ----- fetch -----
        inputs, targets = next(iter(loader))
        inputs = inputs.to(next(model.parameters()).device)
        targets = targets.to(next(model.parameters()).device)
        pred = model(inputs)
        #angle = get_all_octahedral_angles()[0]
        _, Cin, Din, Hin, Win = inputs.shape
        _, Cout, _, _, _ = pred.shape

        rotated_input = rotate_octahedral_exact(inputs, angle, rotate_channels=True, mode="cartesian")
        rotated_pred = rotate_octahedral_exact(pred, angle, rotate_channels=True, mode="cartesian")
        rotated_then_pred = model(rotated_input)

        idx=0
        slice_idx= 7
        fig, axes = plt.subplots(6,Cin,figsize=(30,30))

        for i in range(Cin):
            im=axes[0,i].imshow(inputs[idx][i][:,:,slice_idx].detach().cpu().numpy(), cmap='RdBu_r',origin='lower')
            axes[0,i].set_title(f'Input {i}')
            axes[0,i].axis('off')
            plt.colorbar(im, ax=axes[0, i], fraction=0.046)
        for i in range(Cin):
            axes[1,i].imshow(rotated_input[idx][i][:,:,slice_idx].detach().cpu().numpy(), cmap='RdBu_r',origin='lower')
            axes[1,i].set_title(f'Rotated Input {i}')
            axes[1,i].axis('off')
            plt.colorbar(im, ax=axes[1, i], fraction=0.046)
        for i in range(Cout):
            axes[2,i].imshow(pred[idx][i][:,:,slice_idx].detach().cpu().numpy(), cmap='RdBu_r',origin='lower')
            axes[2,i].set_title(f'Prediction {i}')
            axes[2,i].axis('off')
        for i in range(Cout):
            im=axes[3,i].imshow(rotated_pred[idx][i][:,:,slice_idx].detach().cpu().numpy(), cmap='RdBu_r',origin='lower')
            axes[3,i].set_title(f'Rotated Pred {i}')
            axes[3,i].axis('off')
            plt.colorbar(im, ax=axes[3, i], fraction=0.046)
        for i in range(Cout):
            im=axes[4,i].imshow(rotated_then_pred[idx][i][:,:,slice_idx].detach().cpu().numpy(), cmap='RdBu_r',origin='lower')
            axes[4,i].set_title(f'Rotated Then Pred {i}')
            axes[4,i].axis('off')
            plt.colorbar(im, ax=axes[4, i], fraction=0.046)
        for i in range(Cout):
            im=axes[5,i].imshow(rotated_pred[idx][i][:,:,slice_idx].detach().cpu().numpy() - rotated_then_pred[idx][i][:,:,slice_idx].detach().cpu().numpy(), cmap='RdBu_r',origin='lower')
            axes[5,i].set_title(f'Difference {i}')
            axes[5,i].axis('off')
            plt.colorbar(im, ax=axes[5, i], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()



def visualize_predictions_3D(model, loader, output_path, title, center_mode: str = "cell"):
    """
    center_mode:
      - "cell": coordinates at (i + 0.5)/N  (cell-centered data; default)
      - "node": coordinates at i/(N-1)      (node-centered data)
    """
    def grid_coords(n: int, mode: str):
        if mode == "cell":
            return (np.arange(n) + 0.5) / n
        elif mode == "node":
            if n == 1:
                return np.array([0.0])
            return np.linspace(0.0, 1.0, n)
        else:
            raise ValueError("center_mode must be 'cell' or 'node'")

    def center_index(n: int, mode: str):
        coords = grid_coords(n, mode)
        return int(np.argmin(np.abs(coords - 0.5)))

    def map_index_from_src_to_dst(i_src: int, n_src: int, n_dst: int, mode: str):
        x_src = grid_coords(n_src, mode)[i_src]
        coords_dst = grid_coords(n_dst, mode)
        return int(np.argmin(np.abs(coords_dst - x_src)))
    
    def get_component_labels(C):
        """Get component labels based on number of channels"""
        if C == 1:
            return ['Scalar']
        elif C == 3:
            return ['u', 'v', 'w']
        elif C == 6:
            return ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
        elif C == 9:
            return ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
        else:
            return [f'C{i}' for i in range(C)]
    
    def compute_magnitude(data):
        """Compute magnitude based on number of channels"""
        C = data.shape[0]
        if C == 1:
            return np.abs(data[0])
        elif C == 3:
            return np.sqrt(np.sum(data**2, axis=0))
        elif C == 6:
            return np.sqrt(data[0]**2 + data[1]**2 + data[2]**2 + 
                          2*data[3]**2 + 2*data[4]**2 + 2*data[5]**2)
        elif C == 9:
            return np.sqrt(np.sum(data**2, axis=0))
        else:
            return np.sqrt(np.sum(data**2, axis=0))

    model.eval()
    with torch.no_grad():
        # ----- fetch -----
        inputs, targets = next(iter(loader))
        input_sample  = inputs[0:1].to(next(model.parameters()).device)
        target_sample = targets[0].to(next(model.parameters()).device)
        prediction    = model(input_sample)[0]

        # ----- to numpy -----
        inp  = input_sample[0].detach().cpu().numpy()
        tgt  = target_sample.detach().cpu().numpy()
        pred = prediction.detach().cpu().numpy()

        Cin, Din, Hin, Win = inp.shape
        Ct,  Dt,  Ht,  Wt  = tgt.shape
        Cp,  Dp,  Hp,  Wp  = pred.shape

        # ----- pick slice locations -----
        cin_d, cin_h, cin_w = center_index(Din, center_mode), center_index(Hin, center_mode), center_index(Win, center_mode)
        ctg_d = map_index_from_src_to_dst(cin_d, Din, Dt, center_mode)
        ctg_h = map_index_from_src_to_dst(cin_h, Hin, Ht, center_mode)
        ctg_w = map_index_from_src_to_dst(cin_w, Win, Wt, center_mode)
        cpr_d = map_index_from_src_to_dst(cin_d, Din, Dp, center_mode)
        cpr_h = map_index_from_src_to_dst(cin_h, Hin, Hp, center_mode)
        cpr_w = map_index_from_src_to_dst(cin_w, Din, Wp, center_mode)

        # ----- extract XY slices for all components -----
        inp_xy_slices = inp[:, cin_d, :, :]      # (Cin, Hin, Win)
        tgt_xy_slices = tgt[:, ctg_d, :, :]      # (Ct, Ht, Wt)
        pred_xy_slices = pred[:, cpr_d, :, :]    # (Cp, Hp, Wp)

        # Get component labels
        labels_inp = get_component_labels(Cin)
        labels_tgt = get_component_labels(Ct)
        labels_pred = get_component_labels(Cp)

        # Compute magnitudes
        inp_mag = compute_magnitude(inp_xy_slices)
        tgt_mag = compute_magnitude(tgt_xy_slices)
        pred_mag = compute_magnitude(pred_xy_slices)

        # ----- rotation analysis -----
        angle = torch.tensor([np.pi/2,-np.pi/2,np.pi/2])
        pred_then_rot = rotate_octahedral_exact(prediction, angle, rotate_channels=False, mode="cartesian")
        pred_then_rot_xy = pred_then_rot[:, cpr_d, :, :].cpu().numpy()
        pred_then_rot_mag = compute_magnitude(pred_then_rot_xy)

        rotated_input = rotate_octahedral_exact(input_sample, angle, rotate_channels=True, mode="cartesian")
        rot_then_pred = model(rotated_input)[0].detach().cpu().numpy()
        rot_then_pred_xy = rot_then_pred[:, cpr_d, :, :]
        rot_then_pred_mag = compute_magnitude(rot_then_pred_xy)
        
        # Component-wise differences
        diff_components = rot_then_pred_xy - pred_then_rot_xy
        diff_mag = rot_then_pred_mag - pred_then_rot_mag

        # ----- determine grid layout -----
        # Row 0: Input components + magnitude
        # Row 1: Target components + magnitude  
        # Row 2: Pred components + magnitude
        # Row 3: Pred → Rot components + magnitude
        # Row 4: Rot → Pred components + magnitude
        # Row 5: Difference components + magnitude
        n_cols = max(Cin, Ct, Cp) + 1  # +1 for magnitude
        
        fig, axes = plt.subplots(6, n_cols, figsize=(4*n_cols, 24))
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        # ----- compute vmin/vmax for each component -----
        # For input row - only input
        component_ranges_input = {}
        for c in range(Cin):
            vmax = np.abs(inp_xy_slices[c]).max()
            component_ranges_input[c] = (-vmax, vmax) if inp_xy_slices[c].min() < 0 else (0, vmax)
        
        # For target and prediction rows - combine target AND all prediction variants
        component_ranges_target_pred = {}
        for c in range(max(Ct, Cp)):
            vals = []
            if c < Ct:
                vals.append(np.abs(tgt_xy_slices[c]).max())
            if c < Cp:
                vals.extend([
                    np.abs(pred_xy_slices[c]).max(),
                    np.abs(pred_then_rot_xy[c]).max(),
                    np.abs(rot_then_pred_xy[c]).max()
                ])
            
            if vals:
                vmax = max(vals)
                # Check if any values are negative to determine if we need symmetric range
                has_negative = False
                if c < Ct and tgt_xy_slices[c].min() < 0:
                    has_negative = True
                if c < Cp:
                    if pred_xy_slices[c].min() < 0 or pred_then_rot_xy[c].min() < 0 or rot_then_pred_xy[c].min() < 0:
                        has_negative = True
                
                component_ranges_target_pred[c] = (-vmax, vmax) if has_negative else (0, vmax)
        
        # For difference row
        diff_ranges = {}
        for c in range(Cp):
            vmax = np.abs(diff_components[c]).max()
            diff_ranges[c] = (-vmax, vmax)
        
        # Magnitude ranges
        mag_max = max(inp_mag.max(), tgt_mag.max(), pred_mag.max(), 
                     pred_then_rot_mag.max(), rot_then_pred_mag.max())
        diff_mag_vmax = max(np.abs(diff_mag).max(), 0.01)

        # ----- Row 0: Input components -----
        for c in range(Cin):
            vmin, vmax = component_ranges_input.get(c, (0, 1))
            im = axes[0, c].imshow(inp_xy_slices[c], cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[0, c].set_title(f'Input {labels_inp[c]}')
            axes[0, c].axis('off')
            plt.colorbar(im, ax=axes[0, c], fraction=0.046)
        # Input magnitude
        im_inp_mag = axes[0, Cin].imshow(inp_mag, cmap='viridis', vmin=0, vmax=mag_max)
        axes[0, Cin].set_title('Input |·|')
        axes[0, Cin].axis('off')
        plt.colorbar(im_inp_mag, ax=axes[0, Cin], fraction=0.046)
        # Hide unused
        for c in range(Cin + 1, n_cols):
            axes[0, c].axis('off')

        # ----- Row 1: Target components -----
        for c in range(Ct):
            vmin, vmax = component_ranges_target_pred.get(c, (0, 1))
            im = axes[1, c].imshow(tgt_xy_slices[c], cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[1, c].set_title(f'Target {labels_tgt[c]}')
            axes[1, c].axis('off')
            plt.colorbar(im, ax=axes[1, c], fraction=0.046)
        # Target magnitude
        im_tgt_mag = axes[1, Ct].imshow(tgt_mag, cmap='viridis', vmin=0, vmax=mag_max)
        axes[1, Ct].set_title('Target |·|')
        axes[1, Ct].axis('off')
        plt.colorbar(im_tgt_mag, ax=axes[1, Ct], fraction=0.046)
        # Hide unused
        for c in range(Ct + 1, n_cols):
            axes[1, c].axis('off')

        # ----- Row 2: Prediction components -----
        for c in range(Cp):
            vmin, vmax = component_ranges_target_pred.get(c, (0, 1))
            im = axes[2, c].imshow(pred_xy_slices[c], cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[2, c].set_title(f'Pred {labels_pred[c]}')
            axes[2, c].axis('off')
            plt.colorbar(im, ax=axes[2, c], fraction=0.046)
        # Prediction magnitude
        im_pred_mag = axes[2, Cp].imshow(pred_mag, cmap='viridis', vmin=0, vmax=mag_max)
        axes[2, Cp].set_title('Pred |·|')
        axes[2, Cp].axis('off')
        plt.colorbar(im_pred_mag, ax=axes[2, Cp], fraction=0.046)
        # Hide unused
        for c in range(Cp + 1, n_cols):
            axes[2, c].axis('off')

        # ----- Row 3: Pred → Rot components -----
        for c in range(Cp):
            vmin, vmax = component_ranges_target_pred.get(c, (0, 1))
            im = axes[3, c].imshow(pred_then_rot_xy[c], cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[3, c].set_title(f'Pred→Rot {labels_pred[c]}')
            axes[3, c].axis('off')
            plt.colorbar(im, ax=axes[3, c], fraction=0.046)
        # Magnitude
        im_rot1_mag = axes[3, Cp].imshow(pred_then_rot_mag, cmap='viridis', vmin=0, vmax=mag_max)
        axes[3, Cp].set_title('Pred→Rot |·|')
        axes[3, Cp].axis('off')
        plt.colorbar(im_rot1_mag, ax=axes[3, Cp], fraction=0.046)
        # Hide unused
        for c in range(Cp + 1, n_cols):
            axes[3, c].axis('off')

        # ----- Row 4: Rot → Pred components -----
        for c in range(Cp):
            vmin, vmax = component_ranges_target_pred.get(c, (0, 1))
            im = axes[4, c].imshow(rot_then_pred_xy[c], cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[4, c].set_title(f'Rot→Pred {labels_pred[c]}')
            axes[4, c].axis('off')
            plt.colorbar(im, ax=axes[4, c], fraction=0.046)
        # Magnitude
        im_rot2_mag = axes[4, Cp].imshow(rot_then_pred_mag, cmap='viridis', vmin=0, vmax=mag_max)
        axes[4, Cp].set_title('Rot→Pred |·|')
        axes[4, Cp].axis('off')
        plt.colorbar(im_rot2_mag, ax=axes[4, Cp], fraction=0.046)
        # Hide unused
        for c in range(Cp + 1, n_cols):
            axes[4, c].axis('off')

        # ----- Row 5: Difference components -----
        for c in range(Cp):
            vmin, vmax = diff_ranges.get(c, (-1, 1))
            im = axes[5, c].imshow(diff_components[c], cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[5, c].set_title(f'Diff {labels_pred[c]}')
            axes[5, c].axis('off')
            plt.colorbar(im, ax=axes[5, c], fraction=0.046)
        # Difference magnitude
        im_diff_mag = axes[5, Cp].imshow(diff_mag, cmap='RdBu_r', vmin=-diff_mag_vmax, vmax=diff_mag_vmax)
        axes[5, Cp].set_title('Diff |·|')
        axes[5, Cp].axis('off')
        plt.colorbar(im_diff_mag, ax=axes[5, Cp], fraction=0.046)
        # Hide unused
        for c in range(Cp + 1, n_cols):
            axes[5, c].axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

def plot_spectra(model, loader, output_path, title):
    # TODO: only uses first batch right now
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(loader))
        inputs = inputs.to(next(model.parameters()).device)
        targets = targets.to(next(model.parameters()).device)
        pred = model(inputs)
        error = abs(pred- targets)
        oct_equiv_err_abs, oct_equiv_err_rel = octahedral_equivariance_error(inputs, pred, targets, model, batch_size=1)

        equiv_err_spectra_abs = []
        equiv_err_spectra_rel = []
        vel_spectra_in = []
        vel_spectra_pred = []
        vel_spectra_tgt = []
        error_spectra = []
        for i in tqdm.tqdm(range(oct_equiv_err_abs.shape[0]), desc="Computing spectra"):
            k, spec = compute_isotropic_spectrum(oct_equiv_err_abs[i].cpu().numpy(), Lx=0.0256)
            equiv_err_spectra_abs.append(spec)
            k_rel, spec_rel = compute_isotropic_spectrum(oct_equiv_err_rel[i].cpu().numpy(), Lx=0.0256, tke_normalize=True)
            equiv_err_spectra_rel.append(spec_rel)

        for i in tqdm.tqdm(range(targets.shape[0]), desc="Computing spectra"):
            k_in, spec_in = compute_isotropic_spectrum(inputs[i].cpu().numpy(), Lx=0.0256, tke_normalize=False)
            vel_spectra_in.append(spec_in)
            k_tgt, spec_tgt = compute_isotropic_spectrum(targets[i].cpu().numpy(), Lx=0.0256, tke_normalize=False)
            vel_spectra_tgt.append(spec_tgt)
            k_pred, spec_pred = compute_isotropic_spectrum(pred[i].cpu().numpy(), Lx=0.0256, tke_normalize=False)
            vel_spectra_pred.append(spec_pred)
            k_mse, spec_mse = compute_isotropic_spectrum(error[i].cpu().numpy(), Lx=0.0256, tke_normalize=False)
            error_spectra.append(spec_mse)

        equiv_err_spectra_abs = np.stack(equiv_err_spectra_abs, axis=0)
        equiv_err_spectra_rel = np.stack(equiv_err_spectra_rel, axis=0)
        vel_spectra_in = np.stack(vel_spectra_in, axis=0)
        vel_spectra_tgt = np.stack(vel_spectra_tgt, axis=0)
        vel_spectra_pred = np.stack(vel_spectra_pred, axis=0)
        error_spectra = np.stack(error_spectra, axis=0)
        mean_equiv_err_spectra_abs = np.mean(equiv_err_spectra_abs, axis=0)
        mean_equiv_err_spectra_rel = np.mean(equiv_err_spectra_rel, axis=0)
        mean_vel_spectra_in = np.mean(vel_spectra_in, axis=0)
        mean_vel_spectra_pred = np.mean(vel_spectra_pred, axis=0)
        mean_vel_spectra_tgt = np.mean(vel_spectra_tgt, axis=0)
        mean_error_spectra = np.mean(error_spectra, axis=0)


        fig, axes = plt.subplots(2, 2, figsize=(12, 12),sharex=True)
        axes[0,0].loglog(k, mean_equiv_err_spectra_abs)
        axes[0,0].loglog(k_mse, mean_error_spectra)

        axes[0,0].set_title('Equivariance Error (Absolute)')
        axes[0,0].set_xlabel('Wavenumber')
        axes[0,0].set_ylabel('Normalized Spectrum')
        axes[0,1].loglog(k, mean_equiv_err_spectra_rel)

        axes[0,1].set_title('Equivariance Error (Relative)')
        axes[0,1].set_xlabel('Wavenumber')
        axes[0,1].set_ylabel('Normalized Spectrum')
        axes[1,0].loglog(k_in, mean_vel_spectra_in)
        axes[1,0].set_title('Velocity Spectrum (In, Tgt, Pred)')
        axes[1,0].loglog(k_tgt, mean_vel_spectra_tgt)
        axes[1,0].loglog(k_pred, mean_vel_spectra_pred)
        axes[1,1].loglog(k_mse, mean_error_spectra)
        axes[1,1].set_title('Error Spectrum')
        axes[1,1].set_xlabel('Wavenumber')
        axes[1,1].set_ylabel('Normalized Spectrum')
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def plot_spectrum(k, spec, output_path, title):
    # TODO: only uses first batch right now


    fig, axes = plt.subplots(1, 1, figsize=(12, 12),sharex=True)
    axes.loglog(k, spec)
    axes.set_title('Equivariance Error (Absolute)')
    axes.set_xlabel('Wavenumber')
    axes.set_ylabel('Normalized Spectrum')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def visualize_intermediate_features(model, loader, output_path, title, center_mode: str = "cell"):
    """
    Visualize intermediate features from MultiScaleResidualSR model.
    Shows XY, XZ, and YZ slices for each intermediate output scale.
    
    Args:
        model: MultiScaleResidualSR model
        loader: data loader 
        output_path: where to save the visualization
        title: plot title
        center_mode: "cell" or "node" for coordinate interpretation
    """
    def grid_coords(n: int, mode: str):
        if mode == "cell":
            return (np.arange(n) + 0.5) / n
        elif mode == "node":
            if n == 1:
                return np.array([0.0])
            return np.linspace(0.0, 1.0, n)
        else:
            raise ValueError("center_mode must be 'cell' or 'node'")

    def center_index(n: int, mode: str):
        coords = grid_coords(n, mode)
        return int(np.argmin(np.abs(coords - 0.5)))

    def map_index_from_src_to_dst(i_src: int, n_src: int, n_dst: int, mode: str):
        x_src = grid_coords(n_src, mode)[i_src]
        coords_dst = grid_coords(n_dst, mode)
        return int(np.argmin(np.abs(coords_dst - x_src)))

    model.eval()
    with torch.no_grad():
        # Get first sample
        inputs, targets = next(iter(loader))
        input_sample = inputs[0:1].to(next(model.parameters()).device)   # (1, Cin, Din, Hin, Win)
        
        # Get intermediate features and final prediction
        try:
            final_pred, intermediate_outputs, raw_features = model(input_sample, return_intermediate=True)
        except ValueError:
            # Fallback for models that don't return raw features
            final_pred, intermediate_outputs = model(input_sample, return_intermediate=True)
            raw_features = None
        
        # Convert to numpy
        inp = input_sample[0].detach().cpu().numpy()  # (Cin, Din, Hin, Win)
        
        # Get input dimensions for reference slicing
        Cin, Din, Hin, Win = inp.shape
        cin_d, cin_h, cin_w = center_index(Din, center_mode), center_index(Hin, center_mode), center_index(Win, center_mode)
        
        # Prepare intermediate outputs for visualization
        intermediate_data = {}
        for scale_name, features in intermediate_outputs.items():
            feat_np = features[0].detach().cpu().numpy()  # (C, D, H, W)
            C, D, H, W = feat_np.shape
            
            # Map center indices to this scale
            c_d = map_index_from_src_to_dst(cin_d, Din, D, center_mode)
            c_h = map_index_from_src_to_dst(cin_h, Hin, H, center_mode)
            c_w = map_index_from_src_to_dst(cin_w, Win, W, center_mode)
            
            # Compute magnitude across channels for each slice
            xy_mag = np.sqrt(np.sum(feat_np[:, c_d, :, :]**2, axis=0))  # (H, W)
            xz_mag = np.sqrt(np.sum(feat_np[:, :, c_h, :]**2, axis=0))  # (D, W)  
            yz_mag = np.sqrt(np.sum(feat_np[:, :, :, c_w]**2, axis=0))  # (D, H)
            
            intermediate_data[scale_name] = {
                'xy': xy_mag,
                'xz': xz_mag, 
                'yz': yz_mag,
                'indices': (c_d, c_h, c_w),
                'shape': (D, H, W)
            }
        
        # Also add input for comparison
        xy_mag_inp = np.sqrt(np.sum(inp[:, cin_d, :, :]**2, axis=0))
        xz_mag_inp = np.sqrt(np.sum(inp[:, :, cin_h, :]**2, axis=0))
        yz_mag_inp = np.sqrt(np.sum(inp[:, :, :, cin_w]**2, axis=0))
        
        intermediate_data['input'] = {
            'xy': xy_mag_inp,
            'xz': xz_mag_inp,
            'yz': yz_mag_inp,
            'indices': (cin_d, cin_h, cin_w),
            'shape': (Din, Hin, Win)
        }
        
        # Create figure - one row per scale, 3 columns (XY, XZ, YZ)
        num_scales = len(intermediate_data)
        fig, axes = plt.subplots(num_scales, 3, figsize=(12, 4*num_scales))
        
        # Handle single row case
        if num_scales == 1:
            axes = axes.reshape(1, -1)
        
        # Determine global colormap range
        all_mags = []
        for data in intermediate_data.values():
            all_mags.extend([data['xy'].max(), data['xz'].max(), data['yz'].max()])
        vmin, vmax = 0.0, max(all_mags)
        
        # Plot each scale
        scale_names = ['input'] + [k for k in intermediate_data.keys() if k != 'input']
        for row, scale_name in enumerate(scale_names):
            data = intermediate_data[scale_name]
            d_idx, h_idx, w_idx = data['indices']
            D, H, W = data['shape']
            
            # XY slice
            im_xy = axes[row, 0].imshow(data['xy'], cmap='viridis', vmin=vmin, vmax=vmax)
            axes[row, 0].set_title(f'{scale_name.upper()}: XY (d={d_idx}, {H}×{W})')
            axes[row, 0].axis('off')
            
            # XZ slice  
            im_xz = axes[row, 1].imshow(data['xz'], cmap='plasma', vmin=vmin, vmax=vmax)
            axes[row, 1].set_title(f'{scale_name.upper()}: XZ (h={h_idx}, {D}×{W})')
            axes[row, 1].axis('off')
            
            # YZ slice
            im_yz = axes[row, 2].imshow(data['yz'], cmap='cividis', vmin=vmin, vmax=vmax)
            axes[row, 2].set_title(f'{scale_name.upper()}: YZ (w={w_idx}, {D}×{H})')
            axes[row, 2].axis('off')
        
        # Add colorbar to the last row
        cbar = plt.colorbar(im_yz, ax=axes[-1, :], shrink=0.8, aspect=30)
        cbar.set_label('Feature Magnitude')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def visualize_feature_refinement(model, loader, output_path, title, center_mode: str = "cell"):
    """
    Visualize raw features at each upsampling scale and the final output.
    Shows how features evolve through the upsampling process.
    
    Args:
        model: MultiScaleResidualSR model
        loader: data loader 
        output_path: where to save the visualization
        title: plot title
        center_mode: "cell" or "node" for coordinate interpretation
    """
    def grid_coords(n: int, mode: str):
        if mode == "cell":
            return (np.arange(n) + 0.5) / n
        elif mode == "node":
            if n == 1:
                return np.array([0.0])
            return np.linspace(0.0, 1.0, n)
        else:
            raise ValueError("center_mode must be 'cell' or 'node'")

    def center_index(n: int, mode: str):
        coords = grid_coords(n, mode)
        return int(np.argmin(np.abs(coords - 0.5)))

    def map_index_from_src_to_dst(i_src: int, n_src: int, n_dst: int, mode: str):
        x_src = grid_coords(n_src, mode)[i_src]
        coords_dst = grid_coords(n_dst, mode)
        return int(np.argmin(np.abs(coords_dst - x_src)))

    model.eval()
    with torch.no_grad():
        # Get first sample
        inputs, targets = next(iter(loader))
        input_sample = inputs[0:1].to(next(model.parameters()).device)   # (1, Cin, Din, Hin, Win)
        
        # Get intermediate features and final prediction
        try:
            final_pred, intermediate_outputs, raw_features = model(input_sample, return_intermediate=True)
        except (ValueError, AttributeError):
            print("Model doesn't support raw feature extraction, skipping refinement visualization")
            return
        
        # Convert to numpy
        inp = input_sample[0].detach().cpu().numpy()  # (Cin, Din, Hin, Win)
        target = targets[0].cpu().numpy() if len(targets) > 0 else None
        final_pred_np = final_pred[0].detach().cpu().numpy()
        
        # Get input dimensions for reference slicing
        Cin, Din, Hin, Win = inp.shape
        cin_d = center_index(Din, center_mode)
        
        # Prepare data for each scale
        scale_data = {}
        
        # Add input
        inp_xy = np.sqrt(np.sum(inp[:, cin_d, :, :]**2, axis=0))
        scale_data['input'] = {
            'data': inp_xy,
            'title': f'Input\n{inp_xy.shape}',
            'shape': (Din, Hin, Win)
        }
        
        # Add raw features for each scale
        scale_names = sorted(raw_features.keys(), key=lambda x: int(x.replace('x', '0')) if x != 'lr' else 0)
        for scale_name in scale_names:
            raw_feat = raw_features[scale_name][0].detach().cpu().numpy()  # (hidden_channels, D, H, W)
            C, D, H, W = raw_feat.shape
            
            # Map center index to this scale
            c_d = map_index_from_src_to_dst(cin_d, Din, D, center_mode)
            
            # Compute magnitude across channels for XY slice
            raw_feat_xy = np.sqrt(np.sum(raw_feat[:, c_d, :, :]**2, axis=0))
            
            scale_data[scale_name] = {
                'data': raw_feat_xy,
                'title': f'{scale_name.upper()}\n{H}×{W}, {C}ch',
                'shape': (D, H, W)
            }
        
        # Add final output
        _, D_final, H_final, W_final = final_pred_np.shape
        c_d_final = map_index_from_src_to_dst(cin_d, Din, D_final, center_mode)
        final_xy = np.sqrt(np.sum(final_pred_np[:, c_d_final, :, :]**2, axis=0))
        scale_data['final'] = {
            'data': final_xy,
            'title': f'Final Output\n{final_xy.shape}',
            'shape': (D_final, H_final, W_final)
        }
        
        # Add target if available
        if target is not None:
            _, D_tgt, H_tgt, W_tgt = target.shape
            c_d_tgt = map_index_from_src_to_dst(cin_d, Din, D_tgt, center_mode)
            target_xy = np.sqrt(np.sum(target[:, c_d_tgt, :, :]**2, axis=0))
            scale_data['target'] = {
                'data': target_xy,
                'title': f'Target\n{target_xy.shape}',
                'shape': (D_tgt, H_tgt, W_tgt)
            }
        
        # Create figure
        num_cols = len(scale_data)
        fig, axes = plt.subplots(1, num_cols, figsize=(4*num_cols, 4))
        
        # Handle single column case
        if num_cols == 1:
            axes = [axes]
        
        # Plot each scale
        plot_order = ['input'] + scale_names + ['final'] + (['target'] if target is not None else [])
        
        for col, scale_name in enumerate(plot_order):
            data_info = scale_data[scale_name]
            
            # Use different colormaps for raw features vs outputs
            if scale_name in scale_names:
                # Raw features - use plasma colormap with their own range
                cmap = 'plasma'
                vmin, vmax = 0, data_info['data'].max()
            else:
                # Input, output, target - use viridis with shared range
                cmap = 'viridis' 
                # Calculate shared range for input/output/target
                shared_data = [scale_data[k]['data'] for k in ['input', 'final'] + (['target'] if target is not None else [])]
                vmin, vmax = 0.0, max(d.max() for d in shared_data)
            
            axes[col].imshow(data_info['data'], cmap=cmap, vmin=vmin, vmax=vmax)
            axes[col].set_title(data_info['title'])
            axes[col].axis('off')
        
        plt.suptitle(f'{title}\nShowing XY slices at center depth')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def analyze_kernel_isotropy(model):
    """
    Analyze the isotropy of learned convolutional kernels at each layer/scale.
    
    Args:
        model: MultiScaleResidualSR model
        
    Returns:
        dict: Dictionary with layer names as keys and isotropy metrics as values
    """
    isotropy_metrics = {}
    
    def compute_kernel_isotropy_metrics(kernel):
        """
        Compute isotropy metrics for a single kernel.
        
        Args:
            kernel: torch.Tensor of shape (out_ch, in_ch, D, H, W) or (out_ch, in_ch, H, W)
            
        Returns:
            dict: Various isotropy metrics
        """
        if kernel.dim() == 5:  # 3D kernel (out_ch, in_ch, D, H, W)
            # Compute spatial statistics
            D, H, W = kernel.shape[-3:]
            
            # Center of mass for each output channel
            coords_d = torch.arange(D, dtype=torch.float32).view(D, 1, 1)
            coords_h = torch.arange(H, dtype=torch.float32).view(1, H, 1)  
            coords_w = torch.arange(W, dtype=torch.float32).view(1, 1, W)
            
            # Compute moments for isotropy analysis
            kernel_abs = torch.abs(kernel)  # (out_ch, in_ch, D, H, W)
            
            # Sum over input channels for each output channel
            kernel_summed = torch.sum(kernel_abs, dim=1)  # (out_ch, D, H, W)
            
            # Compute center of mass
            total_mass = torch.sum(kernel_summed, dim=(-3, -2, -1), keepdim=True)  # (out_ch, 1, 1, 1)
            total_mass = torch.clamp(total_mass, min=1e-8)  # Avoid division by zero
            
            com_d = torch.sum(kernel_summed * coords_d, dim=(-3, -2, -1)) / total_mass.squeeze()
            com_h = torch.sum(kernel_summed * coords_h, dim=(-3, -2, -1)) / total_mass.squeeze()
            com_w = torch.sum(kernel_summed * coords_w, dim=(-3, -2, -1)) / total_mass.squeeze()
            
            # Compute second moments (spread)
            var_d = torch.sum(kernel_summed * (coords_d - com_d.view(-1, 1, 1, 1))**2, dim=(-3, -2, -1)) / total_mass.squeeze()
            var_h = torch.sum(kernel_summed * (coords_h - com_h.view(-1, 1, 1, 1))**2, dim=(-3, -2, -1)) / total_mass.squeeze()
            var_w = torch.sum(kernel_summed * (coords_w - com_w.view(-1, 1, 1, 1))**2, dim=(-3, -2, -1)) / total_mass.squeeze()
            
            # Isotropy metrics
            # 1. Variance ratio (how similar are the spreads in each dimension)
            var_mean = (var_d + var_h + var_w) / 3
            var_std = torch.sqrt(((var_d - var_mean)**2 + (var_h - var_mean)**2 + (var_w - var_mean)**2) / 3)
            isotropy_ratio = 1.0 - (var_std / (var_mean + 1e-8))  # 1 = isotropic, 0 = anisotropic
            
            # 2. Directional energy ratio 
            # Compare energy along each axis
            energy_d = torch.sum(kernel_abs**2, dim=(-2, -1))  # Sum over H,W
            energy_h = torch.sum(kernel_abs**2, dim=(-3, -1))  # Sum over D,W  
            energy_w = torch.sum(kernel_abs**2, dim=(-3, -2))  # Sum over D,H
            
            total_energy = torch.sum(kernel_abs**2, dim=(-3, -2, -1), keepdim=True)
            energy_d_norm = torch.sum(energy_d) / torch.sum(total_energy)
            energy_h_norm = torch.sum(energy_h) / torch.sum(total_energy)
            energy_w_norm = torch.sum(energy_w) / torch.sum(total_energy)
            
            energy_uniformity = 1.0 - torch.std(torch.stack([energy_d_norm, energy_h_norm, energy_w_norm]))
            
            return {
                'isotropy_ratio': isotropy_ratio.mean().item(),
                'isotropy_std': isotropy_ratio.std().item(),
                'energy_uniformity': energy_uniformity.item(),
                'var_d': var_d.mean().item(),
                'var_h': var_h.mean().item(), 
                'var_w': var_w.mean().item(),
                'num_filters': kernel.shape[0]
            }
        else:
            # 2D kernel - similar analysis but simpler
            H, W = kernel.shape[-2:]
            kernel_abs = torch.abs(kernel)
            kernel_summed = torch.sum(kernel_abs, dim=1)  # (out_ch, H, W)
            
            # Simple isotropy measure: compare H vs W variance
            var_h = torch.var(torch.sum(kernel_summed, dim=-1))  # Variance along H
            var_w = torch.var(torch.sum(kernel_summed, dim=-2))  # Variance along W
            
            isotropy_ratio = 1.0 - abs(var_h - var_w) / (var_h + var_w + 1e-8)
            
            return {
                'isotropy_ratio': isotropy_ratio.item(),
                'var_h': var_h.item(),
                'var_w': var_w.item(),
                'num_filters': kernel.shape[0]
            }
    
    def aggregate_isotropy_metrics(metrics_list):
        """Aggregate multiple isotropy metrics into a single representative value."""
        if not metrics_list:
            return {}
        
        # Weight by number of filters when averaging
        total_filters = sum(m['num_filters'] for m in metrics_list)
        
        aggregated = {
            'isotropy_ratio': sum(m['isotropy_ratio'] * m['num_filters'] for m in metrics_list) / total_filters,
            'energy_uniformity': sum(m.get('energy_uniformity', 0) * m['num_filters'] for m in metrics_list) / total_filters,
            'num_filters': total_filters,
            'num_layers': len(metrics_list)
        }
        
        # Add variance metrics if available (3D kernels)
        if 'var_d' in metrics_list[0]:
            aggregated.update({
                'var_d': sum(m['var_d'] * m['num_filters'] for m in metrics_list) / total_filters,
                'var_h': sum(m['var_h'] * m['num_filters'] for m in metrics_list) / total_filters,
                'var_w': sum(m['var_w'] * m['num_filters'] for m in metrics_list) / total_filters,
            })
        
        return aggregated
    
    # Analyze by upsampling scale/resolution
    scale_metrics = {}
    
    # Initial conv1 -> 'lr' scale
    if hasattr(model, 'conv1'):
        lr_metrics = [compute_kernel_isotropy_metrics(model.conv1.weight.data.cpu())]
        isotropy_metrics['lr'] = aggregate_isotropy_metrics(lr_metrics)
    
    # Upsample blocks -> x2, x4, x8, etc.
    if hasattr(model, 'upsample_blocks') and hasattr(model, 'upsample_factors'):
        cumulative_scale = 1
        for i, (block, factor) in enumerate(zip(model.upsample_blocks, model.upsample_factors)):
            cumulative_scale *= factor
            scale_name = f'x{cumulative_scale}'
            
            # Collect all conv layers in this upsample block
            block_layer_metrics = []
            for name, layer in block.named_modules():
                if isinstance(layer, (torch.nn.Conv3d, torch.nn.Conv2d)):
                    layer_metrics = compute_kernel_isotropy_metrics(layer.weight.data.cpu())
                    block_layer_metrics.append(layer_metrics)
            
            if block_layer_metrics:
                isotropy_metrics[scale_name] = aggregate_isotropy_metrics(block_layer_metrics)
    
    # Final conv_out -> 'output' scale
    if hasattr(model, 'conv_out'):
        output_metrics = [compute_kernel_isotropy_metrics(model.conv_out.weight.data.cpu())]
        isotropy_metrics['output'] = aggregate_isotropy_metrics(output_metrics)
    
    return isotropy_metrics


def compute_intermediate_octahedral_equivariance_errors(model, loader, batch_size=None):
    """
    Compute octahedral equivariance errors for intermediate features from MultiScaleResidualSR model.
    
    Args:
        model: MultiScaleResidualSR model with return_intermediate capability
        loader: data loader
        batch_size: batch size for equivariance computation (for memory management)
        
    Returns:
        dict: Dictionary with scale names as keys and mean equivariance errors as values
    """
    model.eval()
    equivariance_errors = {}
    
    with torch.no_grad():
        total_errors = {}
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(next(model.parameters()).device)
            targets = targets.to(next(model.parameters()).device)
   
            try:
                final_pred, intermediate_outputs, _ = model(inputs, return_intermediate=True)
            except ValueError:
                final_pred, intermediate_outputs = model(inputs, return_intermediate=True)
            except Exception as e:
                print(f"Unexpected error getting intermediate outputs: {e}")
                raise
            
            for scale_name, intermediate_pred in intermediate_outputs.items():
                def create_model_func(current_scale):
                    def model_func(rotated_inputs):
                        squeeze_output = False
                        if len(rotated_inputs.shape) == 5:  # [24, C, D, H, W] - batch dim squeezed
                            # Add batch dimension back: [24, C, D, H, W] -> [24, 1, C, D, H, W]
                            rotated_inputs = rotated_inputs.unsqueeze(1)
                            squeeze_output = True
                        
                        orig_shape = rotated_inputs.shape  # [24, B, C, D, H, W]
                        num_rotations, batch_size = orig_shape[0], orig_shape[1]
                        reshaped_inputs = rotated_inputs.view(num_rotations * batch_size, *orig_shape[2:])
                        
                        try:
                            _, intermediate_outs, _ = model(reshaped_inputs, return_intermediate=True)
                            result = intermediate_outs[current_scale]
                        except ValueError:
                            print("ValueError in model call, trying without raw features...")
                            _, intermediate_outs = model(reshaped_inputs, return_intermediate=True)
                            result = intermediate_outs[current_scale]
                        except Exception as e:
                            print(f"Error in model_func for {current_scale}: {e}")
                            raise
                        
                        result_shape = result.shape  # [24*B, C_out, D', H', W']
                        final_result = result.view(orig_shape[0], orig_shape[1], *result_shape[1:])
                        
                        if squeeze_output:
                            final_result = final_result.squeeze(1)  # [24, 1, C, D, H, W] -> [24, C, D, H, W]
                        
                        return final_result
                    return model_func
                
                model_func = create_model_func(scale_name)

                error, rel_error = mean_octahedral_equivariance_error(
                    inputs, intermediate_pred, targets, 
                    model_func,
                    batch_size=batch_size
                )                
                if scale_name not in total_errors:
                    total_errors[scale_name] = 0.0
                total_errors[scale_name] += error.item()
            
            num_batches += 1
        
        # Average over all batches
        equivariance_errors = {k: v / num_batches for k, v in total_errors.items()}
    
    return equivariance_errors


def plot_kernel_isotropy(isotropy_over_time, output_path, title="Kernel Isotropy Analysis"):
    """
    Plot kernel isotropy metrics over training epochs.
    
    Args:
        isotropy_over_time: List of dictionaries, where each dict contains layer_name -> isotropy_metrics mappings
        output_path: where to save the plot
        title: plot title
    """
    if not isotropy_over_time:
        return
    
    # Extract layer names from first entry
    layer_names = list(isotropy_over_time[0].keys())
    line_styles = ['-', '--', '-.', ':']
    epochs = range(len(isotropy_over_time))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot isotropy ratio
    for layer_name in layer_names:
        ratios = [epoch_data[layer_name]['isotropy_ratio'] for epoch_data in isotropy_over_time]
        ax1.plot(epochs, ratios, label=layer_name, linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Isotropy Ratio')
    ax1.set_title('Isotropy Ratio (1=isotropic, 0=anisotropic)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # ax1.set_ylim(0, 1)
    
    # Plot variance in each direction (for 3D kernels)
    directions = ['var_d', 'var_h', 'var_w']
    colors = ['red', 'green', 'blue']
    
    for layer_name, line_style in zip(layer_names, line_styles):
        for direction, color in zip(directions, colors):
            if direction in isotropy_over_time[0][layer_name]:
                variances = [epoch_data[layer_name][direction] for epoch_data in isotropy_over_time]
                ax2.plot(epochs, variances, label=f'{layer_name}_{direction}', 
                        color=color, alpha=0.7, linewidth=1, linestyle=line_style)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Variance')
    ax2.set_title('Spatial Variance by Direction')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_intermediate_equivariance_errors(abs_errors_over_time, rel_errors_over_time, output_path, title="Intermediate Octahedral Equivariance Errors"):
    """
    Plot intermediate equivariance errors over training epochs with absolute on left and relative on right.
    
    Args:
        abs_errors_over_time: List of dictionaries with absolute errors, scale_name -> error mappings
        rel_errors_over_time: List of dictionaries with relative errors, scale_name -> error mappings
        output_path: where to save the plot
        title: plot title
    """
    if not abs_errors_over_time and not rel_errors_over_time:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot absolute errors on the left
    if abs_errors_over_time:
        scale_names = list(abs_errors_over_time[0].keys())
        epochs = range(len(abs_errors_over_time))
        
        for scale_name in scale_names:
            errors = [epoch_errors[scale_name] for epoch_errors in abs_errors_over_time]
            ax1.plot(epochs, errors, label=f'{scale_name}', linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Mean Octahedral Equivariance Error (Absolute)')
        ax1.set_title('Absolute Errors')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
    
    # Plot relative errors on the right
    if rel_errors_over_time:
        scale_names = list(rel_errors_over_time[0].keys())
        epochs = range(len(rel_errors_over_time))
        
        for scale_name in scale_names:
            errors = [epoch_errors[scale_name] for epoch_errors in rel_errors_over_time]
            ax2.plot(epochs, errors, label=f'{scale_name}', linewidth=2)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Octahedral Equivariance Error (Relative)')
        ax2.set_title('Relative Errors')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def compute_isotropic_spectrum(field, Lx, Ly=None, Lz=None, tke_normalize=False, spectral_dealias=True, crop_first_bin = True):
    # computes mean spectrum for a bunch of fields
    spectra = []
    for i in tqdm.tqdm(range(field.shape[0]), desc="Computing spectra"):
        k, spec = compute_single_isotropic_spectrum(field[i].cpu().numpy(), Lx=0.0256)
        spectra.append(spec)
    spectra = np.stack(spectra, axis=0)
    mean_spectra = np.mean(spectra, axis=0)
    return k, mean_spectra

def compute_single_isotropic_spectrum(field, Lx, Ly=None, Lz=None, tke_normalize=False, spectral_dealias=True, crop_first_bin = True):
    """
    field: (3, Nx, Ny, Nz) velocity field [m/s]
    Returns:
      k_plot : 1D array of kappa (rad/length)
      E_plot : normalized spectrum (unitless), integrates to 1 over kept bins
    """
    # TODO: fix normalization stuff in this for comparing velocities.
    if Ly is None: 
        Ly = Lx
    if Lz is None:
        Lz = Lx

    C, Nx, Ny, Nz = field.shape

    field_hat = np.fft.fftn(field, axes=(1,2,3), norm='forward') # Forward divides by N
    E3   = 0.5 * np.sum(field_hat * np.conj(field_hat), axis=0).real 

    # Wavenumbers (radians/length)
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=Ly/Ny)
    kz = 2*np.pi*np.fft.fftfreq(Nz, d=Lz/Nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(KX**2 + KY**2 + KZ**2).ravel()

    # Bin setup
    dk = float(min(abs(kx[1]-kx[0]), abs(ky[1]-ky[0]), abs(kz[1]-kz[0])))
    edges   = np.arange(0.0, k_mag.max() + dk, dk)
    centers = 0.5*(edges[1:] + edges[:-1])

    if spectral_dealias:
        # Mimic spectral solver: 2/3 rule per direction
        kx_cut = (2/3) * np.max(np.abs(kx))
        ky_cut = (2/3) * np.max(np.abs(ky))
        kz_cut = (2/3) * np.max(np.abs(kz))
        k_cut_mag = min(kx_cut, ky_cut, kz_cut)  # Conservative: all directions satisfied
        mask = (centers > 0) & (centers <= k_cut_mag)
    else:
        mask = centers > 0

    # Shell sum only for masked bins
    E_k_masked = []
    widths = np.diff(edges)[mask]
    E3_flat = E3.reshape(-1)

    shell, _ = np.histogram(k_mag, bins=edges, weights=E3_flat)
    shell_masked = shell[mask] / widths
    E_k_masked = shell_masked

    # Normalize by masked-bin TKE so ∫ E_norm dk = 1
    if tke_normalize:
        TKE_masked = np.sum(E_k_masked * widths, axis=0)  # (T,)
        E_k_norm = (E_k_masked.T / TKE_masked).T
        E_1D = E_k_norm.mean(axis=0)
    else:
        E_1D = E_k_masked.mean(axis=0)

    if crop_first_bin:
        return centers[mask][1:], E_k_masked[1:]    
    else:
        return centers[mask], E_k_masked

    

def scale_separate_field(field, Lx, Ly=None, Lz=None, tke_normalize=False, spectral_dealias=True, crop_first_bin = True, num_scales=4, return_intermediates=False):
    """
    field: (3, Nx, Ny, Nz) velocity field [m/s]
    Returns:
      field_scales: list of (3, Nx, Ny, Nz) velocity fields, each at a different scale
    """
    # TODO: fix normalization stuff in this for comparing velocities.
    if Ly is None: 
        Ly = Lx
    if Lz is None:
        Lz = Lx

    C, Nx, Ny, Nz = field.shape

    field_hat = np.fft.fftn(field, axes=(1,2,3), norm='forward') # Forward divides by N
    # Wavenumbers (radians/length)
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=Ly/Ny)
    kz = 2*np.pi*np.fft.fftfreq(Nz, d=Lz/Nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(KX**2 + KY**2 + KZ**2).ravel()

    # Bin setup
    dk = float(min(abs(kx[1]-kx[0]), abs(ky[1]-ky[0]), abs(kz[1]-kz[0])))
    edges   = np.arange(0.0, k_mag.max() + dk, dk)
    centers = 0.5*(edges[1:] + edges[:-1])

    if spectral_dealias:
        # Mimic spectral solver: 2/3 rule per direction
        kx_cut = (2/3) * np.max(np.abs(kx))
        ky_cut = (2/3) * np.max(np.abs(ky))
        kz_cut = (2/3) * np.max(np.abs(kz))
        k_cut_mag = min(kx_cut, ky_cut, kz_cut)  # Conservative: all directions satisfied
        mask = (centers > 0) & (centers <= k_cut_mag)
    else:
        mask = centers > 0

    # Band edges (equally spaced in |k|) from 0 to k_max inclusive; num_scales bins
    edges = np.linspace(0.0, float(k_cut_mag), num_scales + 1)

    # Build per-band masks
    band_masks = []
    for i in range(num_scales):
        lo, hi = edges[i], edges[i+1]
        m = (k_mag >= lo) & (k_mag< hi) & mask
        if i == 0 and crop_first_bin:
            # exclude the DC mode from the lowest band
            m = m & (k_mag > 0)
        band_masks.append(m)

    # Ensure the very last edge includes its upper boundary (numerical safety)
    band_masks[-1] = band_masks[-1] | ((k_mag == edges[-1]) & mask)

    # Inverse FFT per band
    fields_scales = []
    for m in band_masks:
        # broadcast mask to (C, Nx, Ny, Nz)
        mh = np.broadcast_to(m[None, ...], field_hat.shape)
        band_hat = np.where(mh, field_hat, 0.0)
        band = np.fft.ifftn(band_hat, axes=(1,2,3)).real  # back to physical space
        fields_scales.append(band)

    if return_intermediates:
        intermediates = {
            'edges': edges,
            'k_max': k_cut_mag,
            'dealias_mask': mask,
            'band_masks': band_masks,
        }
        return fields_scales, intermediates
    return fields_scales