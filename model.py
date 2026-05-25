"""
DPCFFormer: Lightweight Transformer with Diverse Pooling and Cross Feature Fusion
for Hyperspectral Image Classification

This file contains the core model architecture, including PPCF module,
Transformer encoder, and the final classification head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Residual(nn.Module):
    """Residual connection wrapper."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    """LayerNorm before the function."""

    def __init__(self, dim: int, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PPCFModule(nn.Module):
    """
    Pyramid-Pooling and Cross-Feature Fusion Module (PPCF).
    Replaces multi-head self-attention with multi-scale pooling and cross-scale fusion.
    """

    def __init__(self, dim: int, scales: List[int] = None, stride: int = 4):
        """
        Args:
            dim: Input feature dimension (should be 4 * spectral_bands)
            scales: List of pooling scales, default [4, 8, 16, 32]
            stride: Stride for sliding window pooling
        """
        super().__init__()
        self.dim = dim
        self.scales = scales or [4, 8, 16, 32]
        self.stride = stride
        # Verify compatibility
        assert dim % 4 == 0, f"Input dim must be multiple of 4, got {dim}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, dim)
        Returns:
            Fused tensor of same shape (batch, dim)
        """
        batch, d = x.shape
        # Reshape to (batch, 1, dim) for 1D pooling
        x = x.unsqueeze(1)  # (B, 1, D)

        fused = []
        for scale in self.scales:
            # Apply avg pooling with kernel_size=scale, stride=stride
            # Padding: mirror padding to handle boundaries
            padding = (scale - 1) // 2
            pooled = F.avg_pool1d(x, kernel_size=scale, stride=self.stride,
                                  padding=padding, count_include_pad=False)
            # pooled shape: (B, 1, L) where L = ceil(d / stride) approximately
            # Interpolate back to original length d using linear interpolation
            pooled = F.interpolate(pooled, size=d, mode='linear', align_corners=False)
            fused.append(pooled.squeeze(1))  # (B, D)

        # Cross-feature fusion: concatenate along feature dimension? No, we need output dim = d
        # According to paper: four scale-wise features are concatenated in fixed order,
        # but the output should have same length as original sequence? Actually the paper says:
        # "concatenated in fixed order to enable explicit cross-scale information interaction"
        # and the output dimension is still 4C (same as input). So we can sum or average.
        # Here we use simple average of all scales.
        fused_tensor = torch.stack(fused, dim=0)  # (num_scales, B, D)
        output = fused_tensor.mean(dim=0)  # (B, D)
        return output


class PPCFEncoderLayer(nn.Module):
    """Transformer encoder layer with PPCF replacing attention."""

    def __init__(self, dim: int, mlp_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.ppcf = Residual(PreNorm(dim, PPCFModule(dim)))
        self.ff = Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout)))

    def forward(self, x):
        x = self.ppcf(x)
        x = self.ff(x)
        return x


class GroupPooling(nn.Module):
    """Group pooling layer before final classification."""

    def __init__(self, channels: int, num_classes: int, group_scale: int = 4):
        """
        Args:
            channels: Input feature dimension (should be 4 * spectral_bands)
            num_classes: Number of land cover classes
            group_scale: Group size for pooling (stride)
        """
        super().__init__()
        self.group_scale = group_scale
        self.remainder = channels % group_scale
        self.divisible = channels - self.remainder

        if self.remainder == 0:
            self.fc = nn.Linear(channels // group_scale, num_classes)
        else:
            self.fc = nn.Linear(channels // group_scale + 1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, channels)
        Returns:
            Pooled features of shape (batch, group_pooled_dim)
        """
        if self.remainder == 0:
            # Reshape to (B, num_groups, group_scale) and mean over group_scale
            x = x.view(x.shape[0], self.divisible // self.group_scale, self.group_scale)
            x = x.mean(dim=2)
        else:
            xt = torch.empty((x.shape[0], self.divisible // self.group_scale + 1),
                             device=x.device)
            # Process divisible part
            divisible_part = x[:, :self.divisible].view(
                x.shape[0], self.divisible // self.group_scale, self.group_scale
            ).mean(dim=2)
            xt[:, :-1] = divisible_part
            # Process remainder part
            xt[:, -1] = x[:, self.divisible:].mean(dim=1)
            x = xt
        return self.fc(x)


class DPCFFormer(nn.Module):
    """
    DPCFFormer: Lightweight Transformer for Hyperspectral Image Classification.

    The model expects 1D sequence features pre-processed by SCSP and GPCF modules.
    Input shape: (batch, 4 * spectral_bands)
    """

    def __init__(self, num_classes: int, spectral_bands: int, num_encoders: int = 3,
                 mlp_dim: int = 64, dropout: float = 0.0, group_pool_scale: int = 4):
        """
        Args:
            num_classes: Number of target classes
            spectral_bands: Number of spectral bands (C)
            num_encoders: Number of stacked PPCF encoder layers
            mlp_dim: Hidden dimension in feed-forward network
            dropout: Dropout rate
            group_pool_scale: Scale for final group pooling
        """
        super().__init__()
        input_dim = 4 * spectral_bands
        self.encoder_layers = nn.ModuleList([
            PPCFEncoderLayer(dim=input_dim, mlp_dim=mlp_dim, dropout=dropout)
            for _ in range(num_encoders)
        ])
        self.group_pool = GroupPooling(input_dim, num_classes, group_pool_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input 1D sequences of shape (batch, 4*C)
        Returns:
            Logits of shape (batch, num_classes)
        """
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.group_pool(x)
        return x