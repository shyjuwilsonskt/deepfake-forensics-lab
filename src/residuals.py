import cv2
import numpy as np

def extract_highpass_residual(img, kernel_size=(5, 5), sigma=0):
    """
    Extracts the high-pass spatial residual by subtracting a Gaussian blurred version 
    of the image from the original image.
    
    Args:
        img (np.ndarray): The input grayscale image.
        kernel_size (tuple): The size of the Gaussian blur kernel.
        sigma (float): Gaussian kernel standard deviation.
        
    Returns:
        np.ndarray: The high-pass residual as a float32 array.
    """
    # Convert image to float32 to prevent clipping during subtraction
    img_float = img.astype(np.float32)
    
    # Apply Gaussian Blur (Low-Pass Filter)
    blurred = cv2.GaussianBlur(img_float, kernel_size, sigma)
    
    # High-Pass Residual = Original Image - Gaussian Blurred Image
    residual = img_float - blurred
    
    return residual
