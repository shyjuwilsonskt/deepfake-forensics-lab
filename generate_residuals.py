import os
import numpy as np
from src.data_loader import load_and_preprocess_image
from src.residuals import extract_highpass_residual

def process_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
    
    for idx, fname in enumerate(image_files):
        in_path = os.path.join(input_dir, fname)
        img = load_and_preprocess_image(in_path)
        residual = extract_highpass_residual(img)
        
        out_path = os.path.join(output_dir, f"residual_{idx:03d}.npy")
        np.save(out_path, residual)
        print(f"Processed [{idx+1}/{len(image_files)}]: {fname} -> {out_path}")

if __name__ == "__main__":
    process_dataset("data/raw/real", "data/processed/real")
    process_dataset("data/raw/ai", "data/processed/ai")