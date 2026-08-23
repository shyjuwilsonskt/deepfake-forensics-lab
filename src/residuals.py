"""
src/residuals.py
Spatial Pre-Whitening & High-Pass Noise Residual Extraction Operators.
Maintains full backward compatibility with legacy demo notebooks.
"""

import numpy as np
import cv2
from scipy.ndimage import correlate
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 1. Legacy Gaussian Subtraction Operator (Maintained for Backward Compatibility)
# -----------------------------------------------------------------------------

def extract_highpass_residual(img, kernel_size=(5, 5), sigma=0):
    """
    Extracts the high-pass spatial residual by subtracting a Gaussian blurred version 
    of the image from the original image.
    
    Args:
        img (np.ndarray): The input grayscale or RGB image.
        kernel_size (tuple): The size of the Gaussian blur kernel.
        sigma (float): Gaussian kernel standard deviation.
        
    Returns:
        np.ndarray: The high-pass residual as a float32 array.
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    img_float = gray.astype(np.float32)
    
    # Low-pass Gaussian filtering
    blurred = cv2.GaussianBlur(img_float, kernel_size, sigma)
    
    # Residual = Original - LowPass
    residual = img_float - blurred
    return residual


# -----------------------------------------------------------------------------
# 2. Formal Zero-Sum Pre-Whitening Operators (For IEEE SPL Paper)
# -----------------------------------------------------------------------------

def get_highpass_kernel(kernel_type="laplacian_8"):
    """
    Returns a zero-sum spatial high-pass filter kernel.
    Zero-sum ensures complete suppression of stationary/low-frequency image semantics.
    """
    if kernel_type == "laplacian_8":
        kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]], dtype=np.float32)
    elif kernel_type == "laplacian_4":
        kernel = np.array([[ 0, -1,  0],
                           [-1,  4, -1],
                           [ 0, -1,  0]], dtype=np.float32)
    elif kernel_type == "srm_edge":
        kernel = np.array([[ 0,  0,  0],
                           [-1,  2, -1],
                           [ 0,  0,  0]], dtype=np.float32)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")
        
    return kernel


def extract_spatial_residual_np(image, kernel_type="laplacian_8"):
    """
    Extracts high-pass spatial noise residual using discrete convolution.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        gray = image.squeeze(axis=2)
    else:
        gray = image

    gray = gray.astype(np.float32)
    if gray.max() > 1.0:
        gray = gray / 255.0

    kernel = get_highpass_kernel(kernel_type)
    residual = correlate(gray, kernel, mode="reflect")
    return residual


# -----------------------------------------------------------------------------
# 3. PyTorch GPU Batch Module (For High-Throughput Colab Pro Processing)
# -----------------------------------------------------------------------------

class SpatialPreWhitening(torch.nn.Module):
    def __init__(self, kernel_type="laplacian_8"):
        super(SpatialPreWhitening, self).__init__()
        kernel_np = get_highpass_kernel(kernel_type)
        kernel_tensor = torch.from_numpy(kernel_np).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel_tensor)

    def forward(self, x):
        if x.shape[1] == 3:
            r, g, b = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        else:
            gray = x

        gray = gray.float()
        if gray.max() > 1.0:
            gray = gray / 255.0

        padded = F.pad(gray, (1, 1, 1, 1), mode="reflect")
        residual = F.conv2d(padded, self.kernel)
        return residual


# Alias for seamless usage
extract_spatial_residual = extract_spatial_residual_np
