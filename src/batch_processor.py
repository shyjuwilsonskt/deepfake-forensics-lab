import os
import numpy as np
from src.data_loader import load_and_preprocess_image
from src.residuals import extract_highpass_residual

def process_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('jpg', 'png', 'jpeg'))]

    for idx, fname in enumerate(image_files):
        in_path = os.path.join(input_dir, fname)
        img = load_and_preprocess_image(in_path)
        residual = extract_highpass_residual(img)

        out_path = os.path.join(output_dir, f"residual_{idx:03d}.npy")
        np.save(out_path, residual)
        print(f"Processed [{idx+1}/{len(image_files)}]: {fname} -> {out_path}")

if __name__ == "__main__":
    # Define directories as specified in the checklist
    datasets = [
        {"input": "data/raw/real", "output": "data/processed/real"},
        {"input": "data/raw/ai", "output": "data/processed/ai"}
    ]
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for ds in datasets:
        input_dir = os.path.join(project_root, ds["input"])
        output_dir = os.path.join(project_root, ds["output"])
        
        # Only process if the input directory exists
        if os.path.exists(input_dir):
            print(f"Starting batch processing for {ds['input']} -> {ds['output']}")
            process_dataset(input_dir, output_dir)
        else:
            print(f"Input directory does not exist, skipping: {input_dir}")
