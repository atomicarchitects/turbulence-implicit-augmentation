import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Dict


class PixelShuffle3D(nn.Module):
    """
    3D version of PixelShuffle.
    Rearranges elements in a tensor of shape (*, C * r^3, D, H, W) to a tensor of shape (*, C, D * r, H * r, W * r).
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        channels_out = channels // (self.upscale_factor ** 3)
        
        x = x.view(batch_size, channels_out, self.upscale_factor, self.upscale_factor, self.upscale_factor, depth, height, width)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(batch_size, channels_out, depth * self.upscale_factor, height * self.upscale_factor, width * self.upscale_factor)
        
        return x


def get_model(model_config):
    if model_config.model_type == 'PixelShuffleSR':
        return PixelShuffleSR(
            in_channels=model_config.input_channels,
            out_channels=model_config.output_channels,
            hidden_channels=model_config.hidden_channels,
            upscale_factor=model_config.upscale_factor,
            is_3d=model_config.is_3d
        )
    elif model_config.model_type == 'LinearInterpolator':
        return LinearInterpolator(
            upscale_factor=model_config.upscale_factor,
            is_3d=model_config.is_3d
        )
    elif model_config.model_type == 'MultiScaleSR':
        return MultiScaleSR(
            in_channels=model_config.input_channels,
            out_channels=model_config.output_channels,
            hidden_channels=model_config.hidden_channels,
            upscale_factor=model_config.upscale_factor,
            is_3d=model_config.is_3d
        )
    elif model_config.model_type == 'MultiScaleResidualSR':
        return MultiScaleResidualSR(
            in_channels=model_config.input_channels,
            out_channels=model_config.output_channels,
            hidden_channels=model_config.hidden_channels,
            upscale_factor=model_config.upscale_factor,
            is_3d=model_config.is_3d
        )
    elif model_config.model_type == 'SuperResAvg3D':
        return SuperResAvg3D(
            in_channels=model_config.input_channels,
            out_channels=model_config.output_channels,
            channels=model_config.hidden_channels,
            num_layers=3 # Hard-coded to get 8x upsampling
        ) 
    else:
        raise ValueError(f"Model type {model_config.model_type} not supported")

class ESPCN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: int,
            upscale_factor: int,
            is_3d: bool = False,
    ) -> None:
        super(ESPCN, self).__init__()
        out_channels = int(out_channels * (upscale_factor ** 2))


        conv_layer = nn.Conv3d if is_3d else nn.Conv2d
        pixel_shuffle_layer = nn.PixelShuffle if not is_3d else PixelShuffle3D

        self.feature_maps = nn.Sequential(
            conv_layer(in_channels, hidden_channels, (5, 5, 5) if is_3d else (5, 5), (1, 1, 1) if is_3d else (1, 1), (2, 2, 2) if is_3d else (2, 2)),
            nn.Tanh(),
            conv_layer(hidden_channels, hidden_channels, (3, 3, 3) if is_3d else (3, 3), (1, 1, 1) if is_3d else (1, 1), (1, 1, 1) if is_3d else (1, 1)),
            nn.Tanh(),
        )

        self.sub_pixel = nn.Sequential(
            conv_layer(hidden_channels, out_channels, (3, 3, 3) if is_3d else (3, 3), (1, 1, 1) if is_3d else (1, 1), (1, 1, 1) if is_3d else (1, 1)),
            pixel_shuffle_layer(upscale_factor),
        )

        # Initialize weights
        for module in self.modules():
            if isinstance(module, conv_layer):
                nn.init.normal_(module.weight.data,
                                0.0,
                                math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_maps(x)
        x = self.sub_pixel(x)
        x = torch.clamp_(x, -10, 10)
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


class PixelShuffleSR(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 upscale_factor: int,
                 is_3d: bool = False,
                 ) -> None:
        super().__init__()

        conv_layer = nn.Conv3d if is_3d else nn.Conv2d
        pixel_shuffle_layer = nn.PixelShuffle if not is_3d else PixelShuffle3D

        self.conv1 = conv_layer(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        channels_multiplier = upscale_factor**3 if is_3d else upscale_factor**2
        self.conv2 = conv_layer(hidden_channels, hidden_channels * channels_multiplier, kernel_size=3, padding=1)
        self.pixelshuffle = pixel_shuffle_layer(upscale_factor)
        self.conv3 = conv_layer(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.conv1(x))          # (N, hidden_channels, D, H, W) or (N, hidden_channels, H, W)
        x = self.pixelshuffle(self.conv2(x))  # (N, hidden_channels, D*upscale_factor, H*upscale_factor, W*upscale_factor)
        return self.conv3(x)                  # (N, hidden_channels, D*upscale_factor, H*upscale_factor, W*upscale_factor)
    

### New barebones SR conv models

class MultiScaleSR(nn.Module):
    """
    Progressive upsampling (×2, ×2, ... ×rest) to reach `upscale_factor`.
    By default forward(x) -> y. Call forward(x, return_intermediate=True) to also get intermediate features.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 upscale_factor: int,
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
        features_dict: Dict[str, torch.Tensor] = {}
        projections_dict: Dict[str, torch.Tensor] = {}

        x = self.relu(self.conv1(x))
        features_dict['lr'] = x

        cumulative = 1
        for factor, upsample_block in zip(self.upsample_factors, self.upsample_blocks):
            x = upsample_block(x)
            cumulative *= factor
            features_dict[f'x{cumulative}'] = x

        x = self.conv_out(x)

        if return_intermediate:
            projections_dict = {k: self.intermediate_projection_projection(v) for k, v in features_dict.items()}
            return x, features_dict, projections_dict
        else:
            return x
        


class ResConvBlock(nn.Module):
    """2 convs with a skip; no BN; keeps your padding and activations."""
    def __init__(self, C, conv_layer):
        super().__init__()
        self.c1 = conv_layer(C, C, kernel_size=3, padding=1, bias=True, padding_mode='reflect')
        self.c2 = conv_layer(C, C, kernel_size=3, padding=1, bias=True, padding_mode='reflect')
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        y = self.act(self.c1(self.act(x)))
        y = self.c2(y)
        return x + y
        
class MultiScaleResidualSR(nn.Module):
    """
    Progressive upsampling (×2, ×2, ... ×rest) to reach `upscale_factor`.
    Includes residual skip connections from naive resize.
    By default forward(x) -> y. Call forward(x, return_intermediate=True) to also get probes.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 upscale_factor: int,
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
                ResConvBlock(hidden_channels, conv_layer),
            ))
        self.upsample_blocks = nn.ModuleList(upsample_blocks)

        # Output projection
        self.conv_out = conv_layer(hidden_channels, out_channels, kernel_size=3, padding=1, bias=True, padding_mode='reflect')
        
        # Project probes to same output channels 
        self.probe_projection = conv_layer(hidden_channels, out_channels, kernel_size=1, bias=True, padding_mode='reflect')

    def forward(self, x: torch.Tensor, return_intermediate: bool = False) -> torch.Tensor:
        """
        Forward:
            x:  (N,C,D,H,W) if 3D, or (N,C,H,W) if 2D
            return_intermediate: if True, also return feature taps and 1x1 projections per scale.

        Returns:
            y  (and optionally feats, proj) where feats/proj have keys: 'lr', f'x{cum}'
        """
        features_dict: Dict[str, torch.Tensor] = {}
        intermediate_outputs: Dict[str, torch.Tensor] = {}

        x = self.relu(self.conv1(x))
        features_dict['lr'] = x

        cumulative = 1
        for factor, upsample_block in zip(self.upsample_factors, self.upsample_blocks):
            x = upsample_block(x)
            cumulative *= factor
            features_dict[f'x{cumulative}'] = x

        x = self.conv_out(x)

        if return_intermediate:
            intermediate_outputs = {k: self.probe_projection(v) for k, v in features_dict.items()}
            return x, intermediate_outputs, features_dict
        else:
            return x
        

        
### Jeremiah's model

class SuperResAvg3D(nn.Module):
    def __init__(self, in_channels, out_channels, channels, num_layers):
        super().__init__()
        
        # Initial feature extraction
        self.head = nn.Sequential(
            nn.Conv3d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Feature processing blocks
        blocks = []
        for _ in range(4):  # Few residual-like blocks
            blocks.extend([
                nn.Conv3d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(channels),
                nn.ReLU(inplace=True)
            ])
        self.body = nn.Sequential(*blocks)
        
        # Learned upsampling blocks (8→16→32→64)
        self.ups = nn.ModuleList()
        for i in range(num_layers):
            self.ups.append(nn.Sequential(
                nn.ConvTranspose3d(channels, channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(channels),
                nn.ReLU(inplace=True)
            ))
        
        # Final output
        self.tail = nn.Conv3d(channels, out_channels, kernel_size=3, padding=1)
    #Combine all steps for the forward
    def forward(self, x):
        x = self.head(x)
        residual = x
        x = self.body(x)
        x = x + residual  # Skip connection
        
        for up_block in self.ups:
            x = up_block(x)
        
        return self.tail(x)

# class SuperResAvg3D(nn.Module):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  hidden_channels: int,
#                  upscale_factor: int,
#                  is_3d: bool = True) -> None:
#         super().__init__()

#         # Layer factories for 2D/3D
#         conv_layer = nn.Conv3d if is_3d else nn.Conv2d
#         batchnorm_layer = nn.BatchNorm3d if is_3d else nn.BatchNorm2d
#         deconv_layer = nn.ConvTranspose3d if is_3d else nn.ConvTranspose2d

#         # Initial feature extraction
#         self.conv1 = conv_layer(in_channels, hidden_channels, kernel_size=3, padding=1)
#         self.bn1 = batchnorm_layer(hidden_channels)
#         self.relu = nn.ReLU(inplace=True)

#         # Feature processing blocks (light residual-style stack)
#         body_blocks = []
#         for _ in range(4):
#             body_blocks.extend([
#                 conv_layer(hidden_channels, hidden_channels, kernel_size=3, padding=1),
#                 batchnorm_layer(hidden_channels),
#                 nn.ReLU(inplace=True),
#             ])
#         self.body = nn.Sequential(*body_blocks)

#         # Determine number of 2x upsampling stages from upscale_factor (must be power of 2)
#         assert upscale_factor >= 1, "upscale_factor must be >= 1"
#         num_layers = int(math.log2(upscale_factor)) if upscale_factor > 1 else 0
#         if 2 ** num_layers != upscale_factor:
#             raise ValueError("SuperResAvg3D requires power-of-two upscale_factor (e.g., 2, 4, 8, 16)")

#         # Learned upsampling blocks
#         upsample_blocks = []
#         for _ in range(num_layers):
#             upsample_blocks.append(nn.Sequential(
#                 deconv_layer(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
#                 batchnorm_layer(hidden_channels),
#                 nn.ReLU(inplace=True),
#                 conv_layer(hidden_channels, hidden_channels, kernel_size=3, padding=1),
#                 batchnorm_layer(hidden_channels),
#                 nn.ReLU(inplace=True),
#             ))
#         self.upsample_blocks = nn.ModuleList(upsample_blocks)

#         # Final output projection
#         self.conv_out = conv_layer(hidden_channels, out_channels, kernel_size=3, padding=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.relu(self.bn1(self.conv1(x)))
#         residual_features = x
#         x = self.body(x)
#         x = x + residual_features

#         for up_block in self.upsample_blocks:
#             x = up_block(x)

#         return self.conv_out(x)