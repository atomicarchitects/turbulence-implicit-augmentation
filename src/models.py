import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Any, Callable, Dict, List, Sequence, Optional

from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Linear
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import BatchNorm, Gate

from irreps_utils import (
    gradients_to_irreps,
    irreps_to_symmetric_tensor,
    gradient_channels_to_matrix,
    matrix_to_gradient_channels,
    irreps_to_2ndorder_tensor,
    velocity_to_irreps,
    irreps_to_velocity,
)



def get_model(model_config):
    if model_config.model_type == 'LinearInterpolator':
        return LinearInterpolator(
            upscale_factor=model_config.upscale_factor,
            is_3d=model_config.is_3d
        )
    elif model_config.model_type == 'srCNN':
        return srCNN(
            in_channels=model_config.input_channels,
            out_channels=model_config.output_channels,
            hidden_channels=model_config.hidden_channels,
            upscale_factor=model_config.upscale_factor,
            is_3d=model_config.is_3d
        )
    elif model_config.model_type == 'srUpsampleCNN':
        return srUpsampleCNN(
            in_channels=model_config.input_channels,
            out_channels=model_config.output_channels,
            hidden_channels=model_config.hidden_channels,
            upscale_factor=model_config.upscale_factor,
            is_3d=model_config.is_3d
        )
    elif model_config.model_type == 'srECNN':
        return srECNN(
            input_irreps=model_config.input_irreps,
            output_irreps=model_config.output_irreps,
            hidden_irreps=model_config.hidden_irreps,
            kernel_size=model_config.kernel_size,
            is_3d=model_config.is_3d,
            group=model_config.group,
            num_radial_basis=model_config.num_radial_basis,
            lmax=model_config.lmax,
            steps=model_config.steps,
            activation=model_config.activation,
            conv_block_kwargs=getattr(model_config, 'conv_block_kwargs', None),
        )
    elif model_config.model_type == 'srMLP':
        return srMLP(
            in_channels=model_config.input_channels,
            out_channels=model_config.output_channels,
            hidden_channels=model_config.hidden_channels
        )
    elif model_config.model_type == 'sgsCNN':
        return sgsCNN(
            in_channels=model_config.input_channels,
            out_channels=model_config.output_channels,
            hidden_channels=model_config.hidden_channels,
            kernel_size=model_config.kernel_size
        )
    elif model_config.model_type == 'sgsECNN':
        return sgsECNN(
            input_irreps=model_config.input_irreps,
            output_irreps=model_config.output_irreps,
            hidden_irreps=model_config.hidden_irreps,
            kernel_size=model_config.kernel_size,
            num_radial_basis=model_config.num_radial_basis,
            lmax=model_config.lmax,
            diameter=model_config.diameter,
            steps=model_config.steps,
            conv_block_kwargs=model_config.conv_block_kwargs,
        )   
    else:
        raise ValueError(f"Model type {model_config.model_type} not supported")

# Equivariant modules

def _apply_linear(linear: Linear, x: torch.Tensor) -> torch.Tensor:
    """Apply an e3nn Linear module to data with channel dimension in position 1."""
    return linear(x.movedim(1, -1)).movedim(-1, 1)


def _second_order_irreps(channels: int, symmetric: bool) -> o3.Irreps:
    """
    Build irreps describing copies of rank-2 tensors.

    - Full (not-necessarily symmetric) tensors: 0e (trace) + 1o (antisymmetric) + 2e (symmetric traceless)
      => 9 components.
    - Symmetric tensors: 0e + 2e => 6 components.
    """
    block = o3.Irreps("0e + 2e") if symmetric else o3.Irreps("0e + 1o + 2e")
    block_dim = block.dim

    if channels % block_dim != 0:
        tensor_type = "symmetric" if symmetric else "rank-2"
        raise ValueError(
            f"{tensor_type} tensor channels must be a multiple of {block_dim}; got {channels}."
        )

    multiplicity = channels // block_dim
    return block * multiplicity


class ChannelFirstBatchNorm(nn.Module):
    """Apply e3nn BatchNorm to channel-first tensors."""

    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.bn = BatchNorm(irreps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.movedim(1, -1)
        y = self.bn(y)
        return y.movedim(-1, 1)


class ChannelFirstGate(nn.Module):
    """Apply an e3nn Gate module to channel-first tensors."""

    def __init__(self, gate: Gate) -> None:
        super().__init__()
        self.gate = gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.movedim(1, -1)
        y = self.gate(y)
        return y.movedim(-1, 1)


def _identity_activation(x: torch.Tensor) -> torch.Tensor:
    return x


def _parity_key(irrep: o3.Irrep) -> str:
    parity = irrep.p
    if isinstance(parity, str):
        key = parity
    elif hasattr(parity, "value"):
        key = "e" if parity.value == 1 else "o"
    else:
        key = "e" if parity == 1 else "o"
    if key not in ("e", "o"):
        raise ValueError(f"Unsupported parity '{parity}'.")
    return key


def _scalar_activation_map(name: Optional[str]) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    if name is None or name == "identity":
        return {"e": _identity_activation, "o": _identity_activation}
    if name == "relu":
        return {"e": torch.relu, "o": torch.tanh}
    if name == "silu":
        return {"e": F.silu, "o": torch.tanh}
    if name == "tanh":
        return {"e": torch.tanh, "o": torch.tanh}
    raise ValueError(f"Unsupported activation '{name}'.")


def _gate_activation_map() -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    return {"e": torch.sigmoid, "o": torch.tanh}


class EquivConvolution(torch.nn.Module):
    r"""Convolution on voxels using e3nn tensor products.

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        input irreps
    irreps_out : `e3nn.o3.Irreps`
        output irreps
    irreps_sh : `e3nn.o3.Irreps`
        spherical harmonics irreps (typically `o3.Irreps.spherical_harmonics(lmax)`)
    diameter : float
        diameter of the filter in physical units
    num_radial_basis : int
        number of radial basis functions
    steps : tuple of float
        size of the pixel in physical units
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_sh,
        diameter,
        num_radial_basis,
        steps=(1.0, 1.0, 1.0),
        **kwargs,
    ) -> None:
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)

        self.num_radial_basis = int(num_radial_basis)

        # self-connection
        self.sc = Linear(self.irreps_in, self.irreps_out)

        # connection with neighbors
        r = float(diameter) / 2.0

        s = math.floor(r / steps[0])
        x = torch.arange(-s, s + 1.0) * steps[0]

        s = math.floor(r / steps[1])
        y = torch.arange(-s, s + 1.0) * steps[1]

        s = math.floor(r / steps[2])
        z = torch.arange(-s, s + 1.0) * steps[2]

        lattice = torch.stack(
            torch.meshgrid(x, y, z, indexing="ij"), dim=-1
        )  # [x, y, z, R^3]
        self.register_buffer("lattice", lattice)

        if "padding" not in kwargs:
            kwargs["padding"] = tuple(s // 2 for s in lattice.shape[:3])
        self.kwargs = kwargs

        emb = soft_one_hot_linspace(
            x=lattice.norm(dim=-1),
            start=0.0,
            end=r,
            number=self.num_radial_basis,
            basis="smooth_finite",
            cutoff=True,
        )
        self.register_buffer("emb", emb)

        sh = o3.spherical_harmonics(
            l=self.irreps_sh, x=lattice, normalize=True, normalization="component"
        )  # [x, y, z, irreps_sh.dim]

        self.register_buffer("sh", sh)

        self.tp = FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            shared_weights=False,
            compile_left_right=False,
            compile_right=True,
        )

        self.weight = torch.nn.Parameter(
            torch.randn(self.num_radial_basis, self.tp.weight_numel)
        )

    def kernel(self) -> torch.Tensor:
        weight = self.emb @ self.weight
        norm_factor = (
            self.sh.shape[0] * self.sh.shape[1] * self.sh.shape[2]
            if self.sh.numel() > 0
            else 1.0
        )
        weight = weight / norm_factor
        kernel = self.tp.right(self.sh, weight)  # [x, y, z, irreps_in.dim, irreps_out.dim]
        kernel = torch.einsum("xyzio->oixyz", kernel)
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : `torch.Tensor`
            tensor of shape ``(batch, irreps_in.dim, x, y, z)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, irreps_out.dim, x, y, z)``
        """
        sc = _apply_linear(self.sc, x)
        return sc + F.conv3d(x, self.kernel(), **self.kwargs)


class E3NNConvBlock(nn.Module):
    """
    Minimal equivariant block: convolution -> optional batch norm -> gate activation.
    Gate splits scalar and non-scalar irreps so that non-scalars are always gated.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        irreps_sh: o3.Irreps,
        diameter: float,
        num_radial_basis: int,
        steps: Sequence[float] = (1.0, 1.0, 1.0),
        *,
        apply_batchnorm: bool = False,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.irreps_requested = o3.Irreps(irreps_out)
        scalars = [(mul, ir) for mul, ir in self.irreps_requested if ir.l == 0]
        gated = [(mul, ir) for mul, ir in self.irreps_requested if ir.l > 0]
        self.irreps_scalars = o3.Irreps(scalars)
        self.irreps_gated = o3.Irreps(gated)
        self.irreps_gates = (
            o3.Irreps([(mul, o3.Irrep("0e")) for mul, _ in self.irreps_gated])
            if len(self.irreps_gated) > 0
            else o3.Irreps([])
        )
        self.irreps_after_gate = self.irreps_scalars + self.irreps_gated
        if self.irreps_after_gate != self.irreps_requested:
            raise ValueError(
                "E3NNConvBlock expects scalars (l=0) to precede l>0 irreps in irreps_out "
                "so gate output ordering remains consistent."
            )
        self.irreps_gate_input = (
            self.irreps_scalars + self.irreps_gates + self.irreps_gated
        )
        self.conv = EquivConvolution(
            irreps_in=irreps_in,
            irreps_out=self.irreps_gate_input,
            irreps_sh=irreps_sh,
            diameter=diameter,
            num_radial_basis=num_radial_basis,
            steps=steps,
        )
        self.bn = (
            ChannelFirstBatchNorm(self.irreps_gate_input)
            if apply_batchnorm
            else None
        )
        scalar_act = _scalar_activation_map(activation)
        gate_act = _gate_activation_map()
        self.gate = ChannelFirstGate(
            Gate(
                self.irreps_scalars,
                [scalar_act[_parity_key(ir)] for _, ir in self.irreps_scalars],
                self.irreps_gates,
                [gate_act[_parity_key(ir)] for _, ir in self.irreps_gates],
                self.irreps_gated,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.gate(x)
        return x


class EquivariantUpsampleBlock(nn.Module):
    """
    Closely follows srCNN stage: Upsample -> Conv -> Activation -> Conv -> Activation.
    Uses E3NNConvBlock for each conv/activation pair.
    """

    def __init__(
        self,
        irreps: o3.Irreps,
        scale_factor: float,
        irreps_sh: o3.Irreps,
        diameter: float,
        num_radial_basis: int,
        steps: Sequence[float] = (1.0, 1.0, 1.0),
        *,
        apply_batchnorm: bool = False,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="trilinear", align_corners=False)
        self.block1 = E3NNConvBlock(
            irreps_in=irreps,
            irreps_out=irreps,
            irreps_sh=irreps_sh,
            diameter=diameter,
            num_radial_basis=num_radial_basis,
            steps=steps,
            apply_batchnorm=apply_batchnorm,
            activation=activation,
        )
        self.block2 = E3NNConvBlock(
            irreps_in=irreps,
            irreps_out=irreps,
            irreps_sh=irreps_sh,
            diameter=diameter,
            num_radial_basis=num_radial_basis,
            steps=steps,
            apply_batchnorm=apply_batchnorm,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.block1(x)
        x = self.block2(x)
        return x




class LinearInterpolator(nn.Module):
    def __init__(self, upscale_factor, is_3d: bool = False):
        super(LinearInterpolator, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upscale_factor, 
                                    mode='trilinear' if is_3d else 'bilinear', 
                                    align_corners=True)
        self.dummy_param = nn.Parameter(torch.tensor(0.0)) 
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        return x + 0.0 * self.dummy_param


class srCNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 upscale_factor: int = 4,
                 is_3d: bool = False) -> None:
        super().__init__()
        self.is_3d = is_3d
        self.upscale_factor = int(upscale_factor)
        assert self.upscale_factor >= 1, "upscale_factor must be >= 1"

        conv_layer = nn.Conv3d if is_3d else nn.Conv2d
        self.upsample_mode = 'trilinear' if is_3d else 'bilinear'

        # Input feature extraction
        self.conv1 = conv_layer(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True, padding_mode='reflect')
        self.relu = nn.ReLU(inplace=True)

        # Build progressive upsample factors (e.g., 8 -> [2,2,2], 6 -> [2,3], 3 -> [3])
        upsample_factors: list[int] = []
        remaining = self.upscale_factor
        while remaining > 1 and remaining % 2 == 0:
            upsample_factors.append(2)
            remaining //= 2
        if remaining > 1:
            upsample_factors.append(remaining)
        self.upsample_factors = upsample_factors

        # Per-stage upsample + refinement blocks
        upsample_blocks = []
        for factor in self.upsample_factors:
            upsample_blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=factor, mode=self.upsample_mode, align_corners=False),
                conv_layer(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True, padding_mode='reflect'),
                nn.ReLU(inplace=True),
                conv_layer(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True, padding_mode='reflect'),
                nn.ReLU(inplace=True),
            ))
        self.upsample_blocks = nn.ModuleList(upsample_blocks)

        # Output projection
        self.conv_out = conv_layer(hidden_channels, out_channels, kernel_size=3, padding=1, bias=True, padding_mode='reflect')
        
        # Project probes to same output channels 
        self.intermediate_projection = conv_layer(hidden_channels, out_channels, kernel_size=1, bias=True, padding_mode='reflect')

    def forward(self, x: torch.Tensor, return_intermediate: bool = False) -> torch.Tensor:
        """
        Forward:
            x:  (N,C,D,H,W) if 3D, or (N,C,H,W) if 2D
            return_intermediate: if True, also return feature taps and 1x1 projections per scale.

        Returns:
            y  (and optionally feats, proj) where feats/proj have keys: 'lr', f'x{cum}'
        """
        x = self.relu(self.conv1(x))

        cumulative = 1
        for factor, upsample_block in zip(self.upsample_factors, self.upsample_blocks):
            x = upsample_block(x)
            cumulative *= factor

        x = self.conv_out(x)
        return x
    
class srUpsampleCNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 upscale_factor: int = 4,
                 is_3d: bool = False) -> None:
        super().__init__()
        self.is_3d = is_3d
        self.upscale_factor = int(upscale_factor)
        assert self.upscale_factor >= 1, "upscale_factor must be >= 1"

        conv_layer = nn.Conv3d if is_3d else nn.Conv2d
        self.upsample_mode = 'trilinear' if is_3d else 'bilinear'

        # Input feature extraction
        self.conv1 = conv_layer(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True, padding_mode='reflect')
        self.relu = nn.ReLU(inplace=True)

        # Build progressive upsample factors (e.g., 8 -> [2,2,2], 6 -> [2,3], 3 -> [3])
        upsample_factors: list[int] = []
        remaining = self.upscale_factor
        while remaining > 1 and remaining % 2 == 0:
            upsample_factors.append(2)
            remaining //= 2
        if remaining > 1:
            upsample_factors.append(remaining)
        self.upsample_factors = upsample_factors

        # Per-stage upsample + refinement blocks
        upsample_blocks = []
        for factor in self.upsample_factors:
            upsample_blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=factor, mode=self.upsample_mode, align_corners=False),
                conv_layer(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True, padding_mode='reflect'),
                nn.ReLU(inplace=True),
                conv_layer(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True, padding_mode='reflect'),
                nn.ReLU(inplace=True),
            ))
        self.upsample_blocks = nn.ModuleList(upsample_blocks)

        # Output projection
        self.conv_out = conv_layer(hidden_channels, out_channels, kernel_size=3, padding=1, bias=True, padding_mode='reflect')
        
        # Project probes to same output channels 
        self.intermediate_projection = conv_layer(hidden_channels, out_channels, kernel_size=1, bias=True, padding_mode='reflect')

    def forward(self, x: torch.Tensor, return_intermediate: bool = False) -> torch.Tensor:
        """
        Forward:
            x:  (N,C,D,H,W) if 3D, or (N,C,H,W) if 2D
            return_intermediate: if True, also return feature taps and 1x1 projections per scale.

        Returns:
            y  (and optionally feats, proj) where feats/proj have keys: 'lr', f'x{cum}'
        """
        # Upsample input for global residual
        x_up = F.interpolate(x, scale_factor=self.upscale_factor, mode=self.upsample_mode, align_corners=False)
        
        x = self.relu(self.conv1(x))

        cumulative = 1
        for factor, upsample_block in zip(self.upsample_factors, self.upsample_blocks):
            x = upsample_block(x)
            cumulative *= factor

        x = self.conv_out(x)
        return x + x_up

class srECNN(nn.Module):
    def __init__(
        self,
        input_irreps,
        output_irreps,
        hidden_irreps,
        upscale_factor: int,
        is_3d: bool = True,
        group: str = "so3",
        *,
        lmax: int = 3,
        num_radial_basis: int = 6,
        kernel_diameter: float = 3.0,
        steps: Sequence[float] = (1.0, 1.0, 1.0),
        activation: str = "relu",
        conv_block_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if not is_3d:
            raise NotImplementedError(
                "srECNN currently supports only 3D inputs when using e3nn."
            )

        self.is_3d = True
        self.group = group
        self.upscale_factor = int(upscale_factor)
        assert self.upscale_factor >= 1, "upscale_factor must be >= 1"

        self.steps = tuple(float(s) for s in steps)
        self.irreps_in = o3.Irreps(input_irreps)
        if isinstance(hidden_irreps, (str, o3.Irreps)):
            self.irreps_hidden = o3.Irreps(hidden_irreps)
        else:
            raise TypeError("srECNN expects a single hidden_irreps specification (str or Irreps).")
        self.irreps_out = o3.Irreps(output_irreps)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=lmax)
        self.kernel_diameter = float(kernel_diameter)
        self.num_radial_basis = int(num_radial_basis)
        if conv_block_kwargs is None:
            conv_block_kwargs = {}
        self.apply_batchnorm = conv_block_kwargs.get("apply_batchnorm", True)
        self.activation_name = conv_block_kwargs.get("activation", "silu")
        self.input_block = E3NNConvBlock(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_hidden,
            irreps_sh=self.irreps_sh,
            diameter=self.kernel_diameter,
            num_radial_basis=self.num_radial_basis,
            steps=self.steps,
            apply_batchnorm=self.apply_batchnorm,
            activation=self.activation_name,
        )

        # Progressive factors (e.g., 8 -> [2, 2, 2], 6 -> [2, 3])
        upsample_factors: list[int] = []
        remaining = self.upscale_factor
        while remaining > 1 and remaining % 2 == 0:
            upsample_factors.append(2)
            remaining //= 2
        if remaining > 1:
            upsample_factors.append(remaining)
        self.upsample_factors = upsample_factors

        self.upsample_blocks = nn.ModuleList(
            [
                EquivariantUpsampleBlock(
                    irreps=self.irreps_hidden,
                    scale_factor=factor,
                    irreps_sh=self.irreps_sh,
                    diameter=self.kernel_diameter,
                    num_radial_basis=self.num_radial_basis,
                    steps=self.steps,
                    apply_batchnorm=self.apply_batchnorm,
                    activation=self.activation_name,
                )
                for factor in self.upsample_factors
            ]
        )

        self.output_linear = Linear(self.irreps_hidden, self.irreps_out)
        self.intermediate_linear = Linear(self.irreps_hidden, self.irreps_out)

    def forward(self, x: torch.Tensor, return_intermediate: bool = False):
        """
        x:  tensor of shape (N, C, D, H, W)
        If return_intermediate=True: also return hidden feature taps and 1x1 projections per scale.
        """
        hidden = self.input_block(x)

        cumulative = 1
        for factor, stage in zip(self.upsample_factors, self.upsample_blocks):
            hidden = stage(hidden)
            cumulative *= factor

        y = _apply_linear(self.output_linear, hidden)
        return y


# -----------------------------
# SGS models
# -----------------------------


class sgsMLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: Sequence[int]):
        super().__init__()
        if isinstance(layers, int):
            hidden_layers = [layers]
        else:
            hidden_layers = list(layers)

        modules: List[nn.Module] = []
        prev_dim = in_channels
        for hidden_dim in hidden_layers:
            modules.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.layers = nn.ModuleList(modules)
        self.output_layer = nn.Linear(prev_dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)


class sgsCNN(nn.Module):
    def __init__(self, in_channels=9, out_channels=6, hidden_channels: Sequence[int] = (32, 64, 32), kernel_size=3):
        super().__init__()
        padding = kernel_size // 2  # keep same spatial size

        layers: List[nn.Module] = []
        prev_ch = in_channels
        for ch in hidden_channels:
            layers += [
                nn.Conv3d(prev_ch, ch, kernel_size, padding=padding),
                nn.BatchNorm3d(ch),
                nn.ReLU(inplace=True)
            ]
            prev_ch = ch

        layers.append(nn.Conv3d(prev_ch, out_channels, kernel_size, padding=padding))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class sgsECNN(nn.Module):
    def __init__(
        self,
        input_irreps,
        output_irreps,
        hidden_irreps: Optional[Sequence] = None,
        kernel_size: int = 3,
        *,
        num_radial_basis: Optional[int] = None,
        lmax: int = 3,
        steps: Sequence[float] = (1.0, 1.0, 1.0),
        diameter: float = 3.0,
        conv_block_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.kernel_size = int(kernel_size)
        self.kernel_diameter = diameter
        self.steps = tuple(float(s) for s in steps)
        self.apply_batchnorm = conv_block_kwargs["apply_batchnorm"]
        self.activation_name = conv_block_kwargs["activation_name"]
        self.num_radial_basis = num_radial_basis 
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=lmax)

        self.irreps_in = o3.Irreps(input_irreps)
        if isinstance(hidden_irreps, (str, o3.Irreps)):
            self.irreps_hidden = [o3.Irreps(hidden_irreps)]
        else:
            self.irreps_hidden = [o3.Irreps(ir) for ir in hidden_irreps]
        self.irreps_out = o3.Irreps(output_irreps)

        blocks: List[nn.Module] = []
        current_irreps = self.irreps_in
        for hidden_irreps in self.irreps_hidden:
            blocks.append(
                E3NNConvBlock(
                    irreps_in=current_irreps,
                    irreps_out=hidden_irreps,
                    irreps_sh=self.irreps_sh,
                    diameter=self.kernel_diameter,
                    num_radial_basis=self.num_radial_basis,
                    steps=self.steps,
                    apply_batchnorm=self.apply_batchnorm,
                    activation=self.activation_name,
                )
            )
            current_irreps = hidden_irreps

        self.blocks = nn.ModuleList(blocks)
        self.output_linear = Linear(current_irreps, self.irreps_out)

        if output_irreps.strip() == "0e":
            if input_irreps.strip() == "0e":
                self.forward = self.forward_scalar_vector
            elif input_irreps.strip() == "0e + 1o + 2e":
                self.forward = self.forward_grad_to_scalar
        elif output_irreps.strip() == "1o":
            self.forward = self.forward_scalar_vector
        else:
            self.forward = self.forward_sgs
        print(f"Using forward: {self.forward.__name__}")

    def forward_sgs(self, x: torch.Tensor) -> torch.Tensor:
        """
        This task is 2nd order tensor to symmetric second order tensor.
        """
        features = gradients_to_irreps(x)
        for block in self.blocks:
            features = block(features)
        sgs_irreps = _apply_linear(self.output_linear, features)
        return irreps_to_symmetric_tensor(sgs_irreps)

    def forward_grad_to_scalar(self, x: torch.Tensor) -> torch.Tensor:
        """
        This task is 2nd order tensor to scalar.
        """
        features = gradients_to_irreps(x)
        for block in self.blocks:
            features = block(features)
        sgs_irreps = _apply_linear(self.output_linear, features)
        return sgs_irreps

    def forward_scalar_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        This task is scalar to scalar OR vector to vector.
        NOT scalar to vector!
        """
        #features = torch.linalg.norm(x, dim=0, keepdim=True)
        features = x
        for block in self.blocks:
            features = block(features)
        sgs_irreps = _apply_linear(self.output_linear, features)
        return sgs_irreps
    

def rotate_rank2_change_of_basis(matrix: torch.Tensor) -> torch.Tensor:
    """
    Apply fixed change-of-basis R T R^T to rank-2 Cartesian tensors.
    matrix: [B, 3, 3, ...]
    """
    if matrix.ndim < 3:
        raise ValueError(f"Expected at least 3 dims for rank-2 tensor, got shape {matrix.shape}")
    B = matrix.shape[0]
    spatial_shape = matrix.shape[3:]
    # Use reshape to handle non-contiguous inputs safely.
    T = matrix.reshape(B, 3, 3, -1)  # [B, 3, 3, N]
    R = _CHANGE_OF_COORD.to(matrix.device, matrix.dtype)  # [3, 3]
    T_rot = torch.einsum("ij,bjkn,lk->bikn", R, T, R.T)  # R T R^T
    return T_rot.reshape(B, 3, 3, *spatial_shape)

# Change-of-basis matrix implementing y,z,x -> x,y,z (see e3nn docs).
_CHANGE_OF_COORD = torch.tensor(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
)

def rotate_gradients_change_of_basis(gradients: torch.Tensor) -> torch.Tensor:
    """
    Apply change-of-basis to 9-channel gradient tensors, via R T R^T on the
    underlying 3x3 matrices.
    gradients: [B, 9, ...] laid out as in gradient_channels_to_matrix().
    """
    matrix = gradient_channels_to_matrix(gradients)
    matrix_rot = rotate_rank2_change_of_basis(matrix)
    return matrix_to_gradient_channels(matrix_rot)