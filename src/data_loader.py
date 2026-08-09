import cv2
import numpy as np
import os

def load_and_preprocess_image(image_path, target_size=(512, 512)):
    """
    Loads an image from disk, converts to grayscale, and resizes to target dimension.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Read image in grayscale mode (1 channel)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Unable to read image at path: {image_path}")
        
    # Resize to target dimension
    img_resized = cv2.resize(img, target_size)
    
    return img_resized
