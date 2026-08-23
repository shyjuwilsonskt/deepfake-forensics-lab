"""
src/radial_profiling.py
2D Discrete Fourier Transform & Polar Azimuthal Radial Integration.
Computes 1D Power Spectral Profiles S_R(r) for IEEE SPL benchmarks.
"""

import numpy as np
import torch


def compute_radial_profile_np(residual_2d):
    """
    Computes 2D centered power spectrum and azimuthally integrates along concentric
    frequency rings to yield a 1D radial power spectrum profile S_R(r).

    Parameters:
        residual_2d (np.ndarray): 2D high-pass spatial noise residual (H x W).

    Returns:
        np.ndarray: 1D radial spectral array of length min(H, W) // 2.
    """
    H, W = residual_2d.shape
    u0, v0 = H // 2, W // 2
    max_radius = min(u0, v0)

    # 2D FFT and Centering
    F_transform = np.fft.fft2(residual_2d)
    F_shifted = np.fft.fftshift(F_transform)
    P_2d = np.abs(F_shifted) ** 2  # 2D Power Spectrum

    # Polar distance coordinate grid
    y, x = np.indices((H, W))
    r = np.sqrt((x - v0) ** 2 + (y - u0) ** 2).astype(np.int32)

    # Azimuthal integration
    radial_profile = np.zeros(max_radius, dtype=np.float32)
    for radius in range(max_radius):
        mask = (r == radius)
        if np.any(mask):
            radial_profile[radius] = np.mean(P_2d[mask])

    return radial_profile


def compute_radial_profile_batch_torch(residual_batch):
    """
    GPU-accelerated batch azimuthal radial profiling using PyTorch.

    Parameters:
        residual_batch (torch.Tensor): Batch tensor of shape (B, 1, H, W).

    Returns:
        torch.Tensor: Batch of 1D radial profiles of shape (B, max_radius).
    """
    B, C, H, W = residual_batch.shape
    u0, v0 = H // 2, W // 2
    max_radius = min(u0, v0)

    # 2D FFT & Shift
    F_transform = torch.fft.fft2(residual_batch)
    F_shifted = torch.fft.fftshift(F_transform, dim=(-2, -1))
    P_2d = torch.abs(F_shifted) ** 2  # Shape: (B, 1, H, W)

    # Generate radial distance grid on same device
    y = torch.arange(H, device=residual_batch.device) - u0
    x = torch.arange(W, device=residual_batch.device) - v0
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    r = torch.sqrt(grid_x.float() ** 2 + grid_y.float() ** 2).long()

    # Pre-build masks for efficiency
    profiles = []
    for radius in range(max_radius):
        mask = (r == radius)
        if mask.any():
            # Average power across the ring for each image in batch
            ring_power = (P_2d[:, 0, :, :] * mask).sum(dim=(-2, -1)) / mask.sum()
            profiles.append(ring_power.unsqueeze(1))
        else:
            profiles.append(torch.zeros((B, 1), device=residual_batch.device))

    return torch.cat(profiles, dim=1)
