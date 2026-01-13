import os
import torch
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from ..models.unet import build_encoder_resnet50, ResNetBackboneFeatures, UNetFromEncoder

def load_unet(checkpoint_path, device="cuda"):
    """Loads the U-Net with the specific structure from your project."""
    base_resnet = build_encoder_resnet50()
    feat_encoder = ResNetBackboneFeatures(base_resnet)
    model = UNetFromEncoder(feat_encoder, n_classes=1)
    
    # Load weights
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
    
    model.to(device)
    model.eval()
    return model

def run_inference(image_dir, output_dir, model_path_moco=None, model_path_simclr=None):
    """
    Generates comparison images: Original | MoCo-Unet | SimCLR-Unet
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Models
    model_moco = load_unet(model_path_moco, device) if model_path_moco else None
    model_simclr = load_unet(model_path_simclr, device) if model_path_simclr else None
    
    image_files = glob(os.path.join(image_dir, "*.png")) + glob(os.path.join(image_dir, "*.jpg"))
    print(f"Found {len(image_files)} images to process.")

    for img_path in tqdm(image_files, desc="Running Inference"):
        # Preprocessing (Exact match to train_unet.py)
        original_img = Image.open(img_path).convert("RGB")
        img_size = 256
        img_resized = original_img.resize((img_size, img_size), Image.BILINEAR)
        
        img_tensor = np.array(img_resized).astype(np.float32) / 255.0
        img_tensor = np.transpose(img_tensor, (2,0,1))
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device)
        
        # Predictions
        res_list = [np.array(img_resized)] # Start with original image
        
        # Run MoCo Model
        if model_moco:
            with torch.no_grad():
                logits = model_moco(img_tensor)
                mask = torch.sigmoid(logits).cpu().numpy()[0, 0]
                mask = (mask > 0.5).astype(np.uint8) * 255
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                # Add label
                cv2.putText(mask_rgb, "MoCo-UNet", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                res_list.append(mask_rgb)

        # Run SimCLR Model
        if model_simclr:
            with torch.no_grad():
                logits = model_simclr(img_tensor)
                mask = torch.sigmoid(logits).cpu().numpy()[0, 0]
                mask = (mask > 0.5).astype(np.uint8) * 255
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                # Add label
                cv2.putText(mask_rgb, "SimCLR-UNet", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                res_list.append(mask_rgb)
        
        # Concatenate side-by-side
        final_comparison = np.hstack(res_list)
        
        # Save
        base_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_dir, f"compare_{base_name}"), cv2.cvtColor(final_comparison, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    # Example Usage
    run_inference(
        image_dir="data/processed/test_images",
        output_dir="data/results",
        model_path_moco="weights/unet_moco_final.pth",
        model_path_simclr="weights/unet_simclr_final.pth"
    )