import torch
import torch.nn.functional as F
import math


# Euler helpers from e3nn: https://github.com/e3nn/e3nn/blob/0.5.6/e3nn/o3/_rotation.py#L313
def _matrix_x(angle: torch.Tensor) -> torch.Tensor:
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack(
        [
            torch.stack([o, z, z], dim=-1),
            torch.stack([z, c, -s], dim=-1),
            torch.stack([z, s, c], dim=-1),
        ],
        dim=-2,
    )

def _matrix_y(angle: torch.Tensor) -> torch.Tensor:
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack(
        [
            torch.stack([c, z, s], dim=-1),
            torch.stack([z, o, z], dim=-1),
            torch.stack([-s, z, c], dim=-1),
        ],
        dim=-2,
    )

# def _matrix_z(angle: torch.Tensor) -> torch.Tensor:
#     c, s = torch.cos(angle), torch.sin(angle)
#     o = torch.ones_like(c)
#     z = torch.zeros_like(c)
#     row0 = torch.stack([c, -s, z], dim=-1)
#     row1 = torch.stack([s,  c, z], dim=-1)
#     row2 = torch.stack([z,  z, o], dim=-1)
#     return torch.stack([row0, row1, row2], dim=-2)

def euler_angles_to_matrix(angles: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles [α, β, γ] to rotation matrices.
    
    Parameters
    ----------
    angles : torch.Tensor
        Euler angles [α, β, γ] for the SO(3) group, shape (*shape, 3)
    """
    alpha, beta, gamma = angles[..., 0], angles[..., 1], angles[..., 2]
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    return _matrix_y(alpha) @ _matrix_x(beta) @ _matrix_y(gamma)


def rand_so3_angles(num_angles: int):
    """Generates Euler angles [α, β, γ] for the Y-X-Y convention
    that uniformly sample the SO(3) group (proper rotations, no reflections).
    
    The sampling is done according to the Haar measure on SO(3):
    - α, γ are uniformly distributed in [0, 2π)  
    - β is distributed as arccos(uniform[-1, 1]) in [0, π]

    Returns
    -------
    angles : torch.Tensor
        Euler angles [α, β, γ] for the SO(3) group, shape (num_angles, 3)
    """
    alpha = 2 * math.pi * torch.rand(num_angles)
    gamma = 2 * math.pi * torch.rand(num_angles)
    beta = torch.rand(num_angles).mul(2).sub(1).acos()

    angles = torch.stack([alpha, gamma, beta], dim=1)  # Shape: [num_angles, 3]
    return angles


def get_all_octahedral_angles():
    """Generate all Euler angles from the 24-element octahedral group.
    
    The 24 angle sets correspond to:
    - 6 face orientations (which face is "up"): ±x, ±y, ±z
    - 4 rotations around each face: 0°, 90°, 180°, 270°
 
    Returns
    -------
    octahedral_angles : torch.Tensor
        Euler angles [α, β, γ] for the 24-element octahedral group, shape (24, 3)   
    """
    # Using Y-X-Y convention: R = Ry(α) @ Rx(β) @ Ry(γ)
    # Each entry is (alpha, beta, gamma) in radians
    octahedral_euler_angles = [
        # +Z face up (identity orientation) with 0°, 90°, 180°, 270° rotations
        (0, 0, 0),                          # Identity
        (0, 0, math.pi/2),                  # 90° around Z
        (0, 0, math.pi),                    # 180° around Z  
        (0, 0, 3*math.pi/2),               # 270° around Z
        
        # +Y face up (90° around X) with 0°, 90°, 180°, 270° rotations
        (0, math.pi/2, 0),                  # +Y up
        (0, math.pi/2, math.pi/2),          # +Y up + 90° 
        (0, math.pi/2, math.pi),            # +Y up + 180°
        (0, math.pi/2, 3*math.pi/2),       # +Y up + 270°
        
        # -Z face up (180° around X) with 0°, 90°, 180°, 270° rotations  
        (0, math.pi, 0),                    # -Z up
        (0, math.pi, math.pi/2),            # -Z up + 90°
        (0, math.pi, math.pi),              # -Z up + 180°
        (0, math.pi, 3*math.pi/2),         # -Z up + 270°
        
        # -Y face up (270° around X) with 0°, 90°, 180°, 270° rotations
        (0, 3*math.pi/2, 0),               # -Y up  
        (0, 3*math.pi/2, math.pi/2),       # -Y up + 90°
        (0, 3*math.pi/2, math.pi),         # -Y up + 180°
        (0, 3*math.pi/2, 3*math.pi/2),    # -Y up + 270°
        
        # +X face up (90° around Y) with 0°, 90°, 180°, 270° rotations
        (math.pi/2, 0, 0),                  # +X up
        (math.pi/2, 0, math.pi/2),          # +X up + 90°
        (math.pi/2, 0, math.pi),            # +X up + 180°
        (math.pi/2, 0, 3*math.pi/2),       # +X up + 270°
        
        # -X face up (270° around Y) with 0°, 90°, 180°, 270° rotations
        (3*math.pi/2, 0, 0),               # -X up
        (3*math.pi/2, 0, math.pi/2),       # -X up + 90°
        (3*math.pi/2, 0, math.pi),         # -X up + 180°
        (3*math.pi/2, 0, 3*math.pi/2) ,    # -X up + 270°
    ]
    
    octahedral_angles = torch.tensor(octahedral_euler_angles)  # (24, 3)
    return octahedral_angles


def rotate_3d(X: torch.Tensor,
              angles: torch.Tensor,
              rotate_channels: bool = True,
              mode: str = "cartesian") -> torch.Tensor:
    """
    Rotate a 3D field X ([B_x,C,D,H,W]) by Euler angles (alpha, beta, gamma).
    
    Parameters
    ----------
    X : torch.Tensor
        Input tensor of shape [C, D, H, W] (single sample) or [B_x, C, D, H, W] (batch)
    angle : torch.Tensor
        Tensor of Euler angles of shape [3] (single rotation) or [B_angle, 3] (batch)
    rotate_channels : bool, default True
        If True and C==3, also rotate channel vectors by R
    mode : str, default "pairwise"
        - "pairwise": Each angle applied to corresponding X (requires B_x == B_angle)
        - "cartesian": All angles applied to all X samples (B_x * B_angle outputs)
    
    Returns
    -------
    torch.Tensor
        Rotated tensor:
        - Pairwise mode: [C, D, H, W] (if single sample and single rotation) or [B_x, C, D, H, W] 
        - Cartesian mode: [C, D, H, W] (if single sample and single rotation) or [B_x * B_angle, C, D, H, W]
    """
    X_dim = X.ndim
    if X_dim == 4:
        X = X.unsqueeze(0)  # Add batch dimension: [C, D, H, W] -> [1, C, D, H, W]
    elif X_dim != 5:
        raise ValueError(f"Invalid input tensor shape: {X.shape}. Must be either [C, D, H, W] or [B_x, C, D, H, W].")
    B_x, C, D, H, W = X.shape

    if angles.ndim == 1:
        B_angle = 1
        assert angles.shape == (3,), f"Single rotation requires 3 angles, got {angles.shape}. Use euler_angles_to_matrix(angles) to convert to rotation matrix."
    else:
        B_angle = angles.shape[0]
        assert angles.shape == (B_angle, 3), f"Batch of {B_angle} rotations requires {B_angle} angles, got {angles.shape}"
        
    # Generate rotation matrices
    R = euler_angles_to_matrix(angles)  # [B_angle, 3, 3] or [3, 3] if single angle
    if R.ndim == 2:  # Single angle case: [3, 3] -> [1, 3, 3]
        R = R.unsqueeze(0)
    
    if mode == "pairwise": # each angle to corresponding X 
        assert B_x == B_angle, f"Pairwise mode requires matching batch sizes: X={B_x}, angles={B_angle}"
        B_out = B_x
    elif mode == "cartesian": #all angles applied to all X samples
        X = X.unsqueeze(1).expand(-1, B_angle, -1, -1, -1, -1).contiguous().view(B_x * B_angle, C, D, H, W)
        R = R.unsqueeze(0).expand(B_x, -1, -1, -1).contiguous().view(B_x * B_angle, 3, 3)
        B_out = B_x * B_angle
    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'pairwise' or 'cartesian'")
    
    # Apply spatial rotation
    R_inv = R.transpose(-2, -1)  # Inverse for affine grid
    
    # Create affine transformation matrices [B_out, 3, 4]
    A = torch.cat([R_inv, torch.zeros(B_out, 3, 1).to(R_inv.device)], dim=2)
    
    # Apply spatial transformation
    grid = F.affine_grid(A, size=X.size(), align_corners=False).to(X.device)
    X_rot = F.grid_sample(X, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    
    # Apply channel rotation if requested
    if rotate_channels and C == 3:
        X_flat = X_rot.view(B_out, C, -1)  # [B_out, 3, D*H*W]
        X_rot = torch.bmm(R, X_flat).view(B_out, C, D, H, W)
        
    if B_out == 1 and X_dim == 4:
        X_rot = X_rot.squeeze(0)
    
    return X_rot


def apply_octahedral_augmentation(inputs, targets):
    """
    Apply octahedral data augmentation to the inputs and targets.
    """
    octahedral_angles = get_all_octahedral_angles().to(inputs.device) 
    random_angles = octahedral_angles[torch.randint(0, len(octahedral_angles), (inputs.shape[0],))]
    
    rotated_inputs = rotate_3d(inputs, random_angles, rotate_channels=True, mode="pairwise")
    rotated_targets = rotate_3d(targets, random_angles, rotate_channels=True, mode="pairwise")
    return rotated_inputs, rotated_targets

def apply_so3_augmentation(inputs, targets):
    """
    Apply SO(3) data augmentation to the inputs and targets.
    """
    random_angles = rand_so3_angles(inputs.shape[0]).to(inputs.device)

    rotated_inputs = rotate_3d(inputs, random_angles, rotate_channels=True, mode="pairwise")
    rotated_targets = rotate_3d(targets, random_angles, rotate_channels=True, mode="pairwise")
    return rotated_inputs, rotated_targets
