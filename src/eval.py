import torch
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from transformations import rand_so3_angles, get_all_octahedral_angles, rotate_3d

def octahedral_equivariance_error(inputs, pred, targets, model, batch_size=None):
    """
    Returns the pointwise equivariance error for a 3D volume under the 24 rotation symmetries of a cube.

    Equivariance error is |f(g·x) - g·f(x)|
    If relative=True, we normalize this quantity by ???? TODO

    Inputs:
        inputs: [B, C_in, D, H, W]
        pred:   [B, C_out, D', H', W'] produced by model(inputs)
        model:  The model to evaluate. Needed because we'll call it on rotated versions of the input.
        relative: If True, returns the relative equivariance error (see above normalization), otherwise returns the absolute equivariance error.
        batch_size: If provided, process inputs in batches of this size to save memory
        Returns:
        error: [24, B, C, D, H, W]
    """
    B = inputs.shape[0]
    all_angles = get_all_octahedral_angles().to(inputs.device)  # [24, 3]
    
    if batch_size is None or batch_size >= B:
        batch_size = B
    
    error_list = []
    rel_error_list = []
    for start_idx in range(0, B, batch_size):
        end_idx = min(start_idx + batch_size, B)

        inputs_batch = inputs[start_idx:end_idx] 
        pred_batch = pred[start_idx:end_idx]      
        
        rotated_inputs_batch = rotate_3d(inputs_batch, angles=all_angles) 
        pred_then_rotated_batch = rotate_3d(pred_batch, angles=all_angles)
        
        rotated_then_pred_batch = model(rotated_inputs_batch)  
        
        diff = rotated_then_pred_batch - pred_then_rotated_batch
        error_batch = torch.abs(diff)
        error_list.append(error_batch)

        # relative error
        den = pred_then_rotated_batch.abs().mean(dim=(-4,-3,-2,-1), keepdim=True).clamp_min(1e-8)  # [R,B,1,1,1,1]
        rel_error_batch = diff.abs() / den  # [R,B,C,D,H,W]
        rel_error_list.append(rel_error_batch)
                
    # Concatenate along batch dimension: [24, B, C, D, H, W]
    error = torch.cat(error_list, dim=1)
    rel_error = torch.cat(rel_error_list, dim=1)
    return error, rel_error


def mean_octahedral_equivariance_error(inputs, pred, targets, model, batch_size=None):
    absolute_error, relative_error = octahedral_equivariance_error(inputs, pred, targets, model, batch_size=batch_size)
    return torch.mean(absolute_error), torch.mean(relative_error)


def so3_equivariance_error(inputs, pred, targets, model, batch_size=None, num_rotations=100, rotation_batch_size=32):
    """
    Returns the pointwise equivariance error for a 3D volume under random SO(3) rotations.

    Equivariance error is |f(g·x) - g·f(x)| where g is a random SO(3) rotation.
    If relative=True, we normalize this quantity by the magnitude of g·f(x).

    Inputs:
        inputs: [B, C_in, D, H, W]
        pred:   [B, C_out, D', H', W'] produced by model(inputs)
        targets: [B, C_out, D', H', W'] ground truth (not used in computation but kept for API consistency)
        model:  The model to evaluate. Needed because we'll call it on rotated versions of the input.
        relative: If True, returns the relative equivariance error, otherwise returns the absolute equivariance error.
        batch_size: If provided, process inputs in batches of this size to save memory
        num_rotations: Number of random SO(3) rotations to test (default: 100)
        rotation_batch_size: If provided, process rotations in batches of this size to save memory (default: num_rotations)
        
    Returns:
        error: [num_rotations, B, C, D, H, W]
    """
    B = inputs.shape[0]
    # Generate random SO(3) rotation angles
    random_angles = rand_so3_angles(num_rotations).to(inputs.device)  # [num_rotations, 3]
    
    if batch_size is None or batch_size >= B:
        batch_size = B
    
    if rotation_batch_size is None or rotation_batch_size >= num_rotations:
        rotation_batch_size = num_rotations
    
    error_list = []
    rel_error_list = []
    for start_idx in range(0, B, batch_size):
        end_idx = min(start_idx + batch_size, B)

        inputs_batch = inputs[start_idx:end_idx]  # [batch_size, C_in, D, H, W]
        pred_batch = pred[start_idx:end_idx]      # [batch_size, C_out, D', H', W']
        
        # Process rotations in batches to save memory
        rotation_error_list = []
        rotation_rel_error_list = []
        for rot_start_idx in range(0, num_rotations, rotation_batch_size):
            rot_end_idx = min(rot_start_idx + rotation_batch_size, num_rotations)
            
            # Get batch of rotation angles
            angles_batch = random_angles[rot_start_idx:rot_end_idx]  # [rotation_batch_size, 3]
            
            rotated_inputs_batch = rotate_3d(inputs_batch, angles=angles_batch)  # [rotation_batch_size, batch_size, C_in, D, H, W]
            pred_then_rotated_batch = rotate_3d(pred_batch, angles=angles_batch)  # [rotation_batch_size, batch_size, C_out, D', H', W']
            
            rotated_then_pred_batch = model(rotated_inputs_batch)  
            
            diff = rotated_then_pred_batch - pred_then_rotated_batch
            error_batch = torch.abs(diff)
            rotation_error_list.append(error_batch)

            # relative error
            den = pred_then_rotated_batch.abs().mean(dim=(-4,-3,-2,-1), keepdim=True).clamp_min(1e-8)  # [R,B,1,1,1,1]
            rel_error_batch = diff.abs() / den  # [R,B,C,D,H,W]
            rotation_rel_error_list.append(rel_error_batch)

        rotation_error_list = torch.cat(rotation_error_list, dim=0)
        rotation_rel_error_list = torch.cat(rotation_rel_error_list, dim=0)
        error_list.append(rotation_error_list)
        rel_error_list.append(rotation_rel_error_list)
    
    # Concatenate along batch dimension: [num_rotations, B, C, D, H, W]
    error = torch.cat(error_list, dim=1)
    rel_error = torch.cat(rel_error_list, dim=1)
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
    absolute_error, relative_error = so3_equivariance_error(inputs, pred, targets, model, batch_size=batch_size, num_rotations=num_rotations, rotation_batch_size=rotation_batch_size)
    return torch.mean(absolute_error), torch.mean(relative_error)


### DEPRECATED

# def c4_equivariance_error(inputs, pred, model, relative=False): 
#     c4_equiv_err = 0
#     for theta in range(4):
#         pred_then_rotated = torch.rot90(pred,k=theta,dims = (2,3))
#         rotated_then_pred = model(torch.rot90(inputs,k=theta,dims = (2,3)))
#         if relative:
#             eps = 1e-8
#             num_sq = (rotated_then_pred - pred_then_rotated).flatten(start_dim=1).pow(2).sum(dim=1)     
#             den_sq = pred_then_rotated.flatten(start_dim=1).pow(2).sum(dim=1) + eps
#             c4_equiv_err += (num_sq / den_sq).mean().item()   # scalar for this batch
#         else:
#             c4_equiv_err += torch.mean(torch.abs(rotated_then_pred - pred_then_rotated)).item()
#     return c4_equiv_err / 4


# def s4_equivariance_error(inputs, pred, model, relative=False):
#     """
#     Equivariance error for 3D volumes under the 24 rotation symmetries of a cube.

#     inputs: [B, C_in, D, H, W]
#     pred:   [B, C_out, D', H', W'] produced by model(inputs)

#     For each of the 24 rotations g in the cube rotation group:
#       - Compare f(g·x) versus g·f(x)

#     Returns the mean error over the 24 rotations. If relative=True, uses a
#     relative squared error per sample: ||f(g·x) - g·f(x)||^2 / (||g·f(x)||^2 + eps),
#     averaged over the batch, then averaged over the 24 rotations, then sqrt.
#     Otherwise returns mean absolute error averaged over the 24 rotations.
#     """
#     # Helper rotations around principal axes for 5D tensors [B,C,D,H,W]
#     def rot_x(t, k):
#         # rotate around X (width) axis -> rotate (D,H) plane
#         return torch.rot90(t, k=k, dims=(2, 3))

#     def rot_y(t, k):
#         # rotate around Y (height) axis -> rotate (D,W) plane
#         return torch.rot90(t, k=k, dims=(2, 4))

#     def rot_z(t, k):
#         # rotate around Z (depth) axis -> rotate (H,W) plane
#         return torch.rot90(t, k=k, dims=(3, 4))

#     # Channel rotation for vector fields (C == 3) under 90-degree steps
#     def ch_rot_x(t, k):
#         if t.shape[1] != 3:
#             return t
#         k = k % 4
#         if k == 0:
#             return t
#         vx, vy, vz = t[:, 0:1], t[:, 1:2], t[:, 2:3]
#         if k == 1:   # rotation in (z,y) plane: (z,y)->(-y,z) => (x, y, z) -> (x, z, -y)
#             return torch.cat([vx, vz, -vy], dim=1)
#         if k == 2:   # y' = -y, z' = -z, x' = x
#             return torch.cat([vx, -vy, -vz], dim=1)
#         # k == 3: inverse of k==1: (x, y, z) -> (x, -z, y)
#         return torch.cat([vx, -vz, vy], dim=1)

#     def ch_rot_y(t, k):
#         if t.shape[1] != 3:
#             return t
#         k = k % 4
#         if k == 0:
#             return t
#         vx, vy, vz = t[:, 0:1], t[:, 1:2], t[:, 2:3]
#         if k == 1:   # x' = z, z' = -x, y' = y
#             return torch.cat([vz, vy, -vx], dim=1)
#         if k == 2:   # x' = -x, z' = -z, y' = y
#             return torch.cat([-vx, vy, -vz], dim=1)
#         # k == 3:    # x' = -z, z' = x, y' = y
#         return torch.cat([-vz, vy, vx], dim=1)

#     def ch_rot_z(t, k):
#         if t.shape[1] != 3:
#             return t
#         k = k % 4
#         if k == 0:
#             return t
#         vx, vy, vz = t[:, 0:1], t[:, 1:2], t[:, 2:3]
#         if k == 1:   # x' = -y, y' = x, z' = z
#             return torch.cat([-vy, vx, vz], dim=1)
#         if k == 2:   # x' = -x, y' = -y, z' = z
#             return torch.cat([-vx, -vy, vz], dim=1)
#         # k == 3:    # x' = y, y' = -x, z' = z
#         return torch.cat([vy, -vx, vz], dim=1)

#     # 6 base orientations for the "up" axis: z, y, -z, -y, x, -x
#     base_transforms = [
#         (lambda t: t,              lambda t: t),             # z up
#         (lambda t: rot_x(t, 1),    lambda t: ch_rot_x(t, 1)),# y up
#         (lambda t: rot_x(t, 2),    lambda t: ch_rot_x(t, 2)),# -z up
#         (lambda t: rot_x(t, 3),    lambda t: ch_rot_x(t, 3)),# -y up
#         (lambda t: rot_y(t, 1),    lambda t: ch_rot_y(t, 1)),# x up
#         (lambda t: rot_y(t, 3),    lambda t: ch_rot_y(t, 3)),# -x up
#     ]

#     total_err = 0.0
#     num_rots = 0

#     for base_spatial, base_channel in base_transforms:
#         inputs_b = base_spatial(inputs)
#         inputs_b = base_channel(inputs_b)
#         pred_b = base_spatial(pred)
#         pred_b = base_channel(pred_b)
#         for k in range(4):
#             pred_then_rotated = rot_z(pred_b, k)
#             pred_then_rotated = ch_rot_z(pred_then_rotated, k)

#             rotated_input = rot_z(inputs_b, k)
#             rotated_input = ch_rot_z(rotated_input, k)
#             rotated_then_pred = model(rotated_input)
#             if relative:
#                 eps = 1e-8
#                 num_sq = (rotated_then_pred - pred_then_rotated).flatten(start_dim=1).pow(2).sum(dim=1)
#                 den_sq = pred_then_rotated.flatten(start_dim=1).pow(2).sum(dim=1) + eps
#                 total_err += (num_sq / den_sq).mean().item()
#             else:
#                 total_err += torch.mean(torch.abs(rotated_then_pred - pred_then_rotated)).item()
#             num_rots += 1

#     return total_err / num_rots  # divide by 24


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
            pred_then_rotated = torch.rot90(predictions[i],k=1,dims=(1,2))[0].cpu().numpy()
            rotated_then_predicted = model(torch.rot90(inputs[i].unsqueeze(0),k=1,dims=(2,3)))[0][0].cpu().numpy()
            
            # Compute difference for diverging colormap
            diff = rotated_then_predicted - pred_then_rotated
            
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

def visualize_predictions_3D(model, loader, output_path, title, center_mode: str = "cell"):
    """
    center_mode:
      - "cell": coordinates at (i + 0.5)/N  (cell-centered data; default)
      - "node": coordinates at i/(N-1)      (node-centered data)
    """
    def grid_coords(n: int, mode: str):
        if mode == "cell":
            # cell centers in [0,1)
            return (np.arange(n) + 0.5) / n
        elif mode == "node":
            if n == 1:
                return np.array([0.0])
            return np.linspace(0.0, 1.0, n)
        else:
            raise ValueError("center_mode must be 'cell' or 'node'")

    def center_index(n: int, mode: str):
        # physical midpoint x*=0.5; choose the nearest index
        coords = grid_coords(n, mode)
        return int(np.argmin(np.abs(coords - 0.5)))

    def map_index_from_src_to_dst(i_src: int, n_src: int, n_dst: int, mode: str):
        # take src index's physical coordinate, pick nearest plane in dst
        x_src = grid_coords(n_src, mode)[i_src]
        coords_dst = grid_coords(n_dst, mode)
        return int(np.argmin(np.abs(coords_dst - x_src)))

    model.eval()
    with torch.no_grad():
        # ----- fetch -----
        inputs, targets = next(iter(loader))
        input_sample  = inputs[0:1].to(next(model.parameters()).device)   # (1, Cin, Din, Hin, Win)
        target_sample = targets[0].to(next(model.parameters()).device)    # (Ct, Dt, Ht, Wt)
        prediction    = model(input_sample)[0]                 # (Cp, Dp, Hp, Wp)

        # ----- to numpy -----
        inp  = input_sample[0].detach().cpu().numpy()          # (Cin, Din, Hin, Win)
        tgt  = target_sample.detach().cpu().numpy()            # (Ct,  Dt,  Ht,  Wt)
        pred = prediction.detach().cpu().numpy()               # (Cp,  Dp,  Hp,  Wp)

        # sanity
        assert pred.shape[0] == 3 and tgt.shape[0] == 3, "Expected 3 channels (u,v,w) for pred and target."
        # input may be 3 (u,v,w) or something else; we’ll still compute magnitude across channels.
        Cin, Din, Hin, Win = inp.shape
        Ct,  Dt,  Ht,  Wt  = tgt.shape
        Cp,  Dp,  Hp,  Wp  = pred.shape

        # ----- pick the same physical slice location(s) based on the INPUT'S center -----
        cin_d, cin_h, cin_w = center_index(Din, center_mode), center_index(Hin, center_mode), center_index(Win, center_mode)
        # map that same physical location to target/pred grids
        ctg_d = map_index_from_src_to_dst(cin_d, Din, Dt, center_mode)
        ctg_h = map_index_from_src_to_dst(cin_h, Hin, Ht, center_mode)
        ctg_w = map_index_from_src_to_dst(cin_w, Win, Wt, center_mode)

        cpr_d = map_index_from_src_to_dst(cin_d, Din, Dp, center_mode)
        cpr_h = map_index_from_src_to_dst(cin_h, Hin, Hp, center_mode)
        cpr_w = map_index_from_src_to_dst(cin_w, Win, Wp, center_mode)

        # ----- magnitudes (aligned in physical space) -----
        # Inputs
        xy_mag_inp = np.sqrt(np.sum(inp[:, cin_d, :, :]**2, axis=0))   # (Hin, Win)
        xz_mag_inp = np.sqrt(np.sum(inp[:, :, cin_h, :]**2, axis=0))   # (Din, Win)
        yz_mag_inp = np.sqrt(np.sum(inp[:, :, :, cin_w]**2, axis=0))   # (Din, Hin)

        # Targets (mapped indices)
        xy_mag_tgt = np.sqrt(np.sum(tgt[:, ctg_d, :, :]**2, axis=0))   # (Ht, Wt)
        xz_mag_tgt = np.sqrt(np.sum(tgt[:, :, ctg_h, :]**2, axis=0))   # (Dt, Wt)
        yz_mag_tgt = np.sqrt(np.sum(tgt[:, :, :, ctg_w]**2, axis=0))   # (Dt, Ht)

        # Predictions (mapped indices)
        xy_mag_pred = np.sqrt(np.sum(pred[:, cpr_d, :, :]**2, axis=0)) # (Hp, Wp)
        xz_mag_pred = np.sqrt(np.sum(pred[:, :, cpr_h, :]**2, axis=0)) # (Dp, Wp)
        yz_mag_pred = np.sqrt(np.sum(pred[:, :, :, cpr_w]**2, axis=0)) # (Dp, Hp)

        # ----- rotation analysis (prediction index space only) -----
        pred_then_rot = torch.rot90(prediction, k=1, dims=(2, 3))        # rotate (H,W)
        pred_then_rot_xy = pred_then_rot[:, cpr_d, :, :].cpu().numpy()
        pred_then_rot_mag = np.sqrt(np.sum(pred_then_rot_xy**2, axis=0))

        rotated_input = torch.rot90(input_sample, k=1, dims=(3, 4))      # rotate (H,W) in input
        rot_then_pred = model(rotated_input)[0].detach().cpu().numpy()   # predict → pred grid
        rot_then_pred_xy = rot_then_pred[:, cpr_d, :, :]
        rot_then_pred_mag = np.sqrt(np.sum(rot_then_pred_xy**2, axis=0))

        diff = rot_then_pred_mag - pred_then_rot_mag

        # ----- figure (4x3) -----
        fig, axes = plt.subplots(4, 3, figsize=(12, 16))

        vmin = 0.0
        vmax = max(
            xy_mag_inp.max(), xz_mag_inp.max(), yz_mag_inp.max(),
            xy_mag_tgt.max(), xz_mag_tgt.max(), yz_mag_tgt.max(),
            xy_mag_pred.max(), xz_mag_pred.max(), yz_mag_pred.max(),
            pred_then_rot_mag.max(), rot_then_pred_mag.max()
        )
        diff_vmax = max(np.abs(diff).max(), 0.1)

        # Row 0: Inputs
        axes[0,0].imshow(xy_mag_inp, cmap='viridis', vmin=vmin, vmax=vmax); axes[0,0].set_title('XY Input'); axes[0,0].axis('off')
        axes[0,1].imshow(xz_mag_inp, cmap='plasma', vmin=vmin, vmax=vmax);  axes[0,1].set_title('XZ Input'); axes[0,1].axis('off')
        axes[0,2].imshow(yz_mag_inp, cmap='cividis', vmin=vmin, vmax=vmax); axes[0,2].set_title('YZ Input'); axes[0,2].axis('off')

        # Row 1: Targets (physically aligned to input center)
        axes[1,0].imshow(xy_mag_tgt, cmap='viridis', vmin=vmin, vmax=vmax); axes[1,0].set_title(f'XY True (d={ctg_d})'); axes[1,0].axis('off')
        axes[1,1].imshow(xz_mag_tgt, cmap='plasma', vmin=vmin, vmax=vmax);  axes[1,1].set_title(f'XZ True (h={ctg_h})'); axes[1,1].axis('off')
        axes[1,2].imshow(yz_mag_tgt, cmap='cividis', vmin=vmin, vmax=vmax); axes[1,2].set_title(f'YZ True (w={ctg_w})'); axes[1,2].axis('off')

        # Row 2: Predictions (physically aligned to input center)
        axes[2,0].imshow(xy_mag_pred, cmap='viridis', vmin=vmin, vmax=vmax); axes[2,0].set_title(f'XY Pred (d={cpr_d})'); axes[2,0].axis('off')
        axes[2,1].imshow(xz_mag_pred, cmap='plasma', vmin=vmin, vmax=vmax);  axes[2,1].set_title(f'XZ Pred (h={cpr_h})'); axes[2,1].axis('off')
        axes[2,2].imshow(yz_mag_pred, cmap='cividis', vmin=vmin, vmax=vmax); axes[2,2].set_title(f'YZ Pred (w={cpr_w})'); axes[2,2].axis('off')

        # Row 3: Rotation analysis (prediction space)
        axes[3,0].imshow(pred_then_rot_mag, cmap='YlGnBu', vmin=vmin, vmax=vmax); axes[3,0].set_title('Pred → Rot 90°'); axes[3,0].axis('off')
        axes[3,1].imshow(rot_then_pred_mag, cmap='YlGnBu', vmin=vmin, vmax=vmax); axes[3,1].set_title('Rot 90° → Pred'); axes[3,1].axis('off')
        im_diff = axes[3,2].imshow(diff, cmap='RdBu_r', vmin=-diff_vmax, vmax=diff_vmax); axes[3,2].set_title('Difference'); axes[3,2].axis('off')

        cbar = plt.colorbar(im_diff, ax=axes[3,2], shrink=0.8); cbar.set_label('Difference')
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
        oct_equiv_err_abs, oct_equiv_err_rel = octahedral_equivariance_error(inputs, pred, targets, model)

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