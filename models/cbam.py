"""
CBAM: Convolutional Block Attention Module (3D)
Adapted for 3D medical imaging from:
Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018

This implementation provides 3D-compatible channel and spatial attention
for use with volumetric CT data.
"""

import torch
import torch.nn as nn


class ChannelAttention3D(nn.Module):
    """
    3D Channel Attention Module.
    
    Applies channel-wise attention by computing average-pooled and max-pooled
    features, then passing through a shared MLP to generate attention weights.
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """
        Args:
            in_channels: Number of input channels
            reduction_ratio: Channel reduction ratio for the bottleneck
        """
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, D, H, W)
            
        Returns:
            Channel attention weights (B, C, 1, 1, 1)
        """
        B, C, D, H, W = x.shape
        
        # Average pooling branch
        avg_out = self.avg_pool(x).view(B, C)
        avg_out = self.mlp(avg_out)
        
        # Max pooling branch
        max_out = self.max_pool(x).view(B, C)
        max_out = self.mlp(max_out)
        
        # Combine and apply sigmoid
        attention = torch.sigmoid(avg_out + max_out)
        
        return attention.view(B, C, 1, 1, 1)


class SpatialAttention3D(nn.Module):
    """
    3D Spatial Attention Module.
    
    Computes attention across spatial dimensions using channel-wise
    average and max pooling followed by a convolution.
    """
    
    def __init__(self, kernel_size: int = 7):
        """
        Args:
            kernel_size: Size of the convolution kernel
        """
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv = nn.Conv3d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, D, H, W)
            
        Returns:
            Spatial attention weights (B, 1, D, H, W)
        """
        # Channel-wise average and max
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(concat))
        
        return attention


class CBAM3D(nn.Module):
    """
    Complete CBAM module for 3D data.
    
    Applies channel attention followed by spatial attention sequentially.
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7
    ):
        """
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for channel attention MLP
            spatial_kernel_size: Kernel size for spatial attention conv
        """
        super().__init__()
        
        self.channel_attention = ChannelAttention3D(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention3D(spatial_kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, D, H, W)
            
        Returns:
            Attention-refined tensor (B, C, D, H, W)
        """
        # Channel attention
        x = x * self.channel_attention(x)
        
        # Spatial attention
        x = x * self.spatial_attention(x)
        
        return x


if __name__ == "__main__":
    # Quick test
    x = torch.randn(2, 64, 32, 32, 32)
    cbam = CBAM3D(in_channels=64)
    out = cbam(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"CBAM parameters: {sum(p.numel() for p in cbam.parameters()):,}")
