import os
import json
from glob import glob
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# Project imports
from ..models.unet import build_encoder_resnet50, ResNetBackboneFeatures, UNetFromEncoder

# ====== Dataset (Exact Copy) ======
class SegTileDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=256):
        self.images = image_paths
        self.masks = mask_paths
        assert len(self.images) == len(self.masks), "images and masks counts differ"
        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Exact logic from notebook
        img = Image.open(self.images[idx]).convert("RGB")
        msk = Image.open(self.masks[idx]).convert("L")
        
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        msk = msk.resize((self.img_size, self.img_size), Image.NEAREST)

        img = np.array(img).astype(np.float32) / 255.0
        msk = np.array(msk).astype(np.float32) / 255.0
        msk = (msk > 0.5).astype(np.float32)

        img = np.transpose(img, (2,0,1))
        msk = np.expand_dims(msk, 0)
        return torch.from_numpy(img), torch.from_numpy(msk)

# ====== Metrics / Loss (Exact Copy) ======
def dice_coef(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + eps) / (pred.sum() + target.sum() + eps)

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dice_loss = 1 - dice_coef(probs, targets)
        return bce_loss + dice_loss

# ====== Training / Validation Loops ======
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    it = 0
    for imgs, masks in tqdm(loader, desc="Train"):
        imgs = imgs.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            d = dice_coef(probs, masks).item()
        total_dice += d
        it += 1
    if it == 0: return 0.0, 0.0
    return total_loss/it, total_dice/it

def valid_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    it = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Valid"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            d = dice_coef(probs, masks).item()
            total_dice += d
            it += 1
    if it==0: return 0.0, 0.0
    return total_loss/it, total_dice/it

def train_unet(data_dir, encoder_weights_path, save_dir, epochs_freeze=5, epochs_finetune=20, batch_size=8, lr=1e-4):
    # --- Fixed settings from notebook ---
    IMG_SIZE = 256
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(save_dir, exist_ok=True)

    # 1. Setup data paths
    images_root_dir = os.path.join(data_dir, "ORIGINAL")
    masks_root_dir = os.path.join(data_dir, "annotated_images")

    all_imgs = sorted(glob(os.path.join(images_root_dir, "*")))
    all_msks = sorted(glob(os.path.join(masks_root_dir, "*")))
    
    if len(all_imgs) == 0:
        print(f"Error: No images found in {images_root_dir}")
        return

    # 2. Split (Seed 42 for reproducibility)
    n = len(all_imgs)
    ntr = int(0.8 * n)
    indices = list(range(n))
    random.seed(42) 
    random.shuffle(indices)
    
    train_indices = indices[:ntr]
    val_indices = indices[ntr:]
    
    train_imgs = [all_imgs[i] for i in train_indices]
    val_imgs = [all_imgs[i] for i in val_indices]
    train_msks = [all_msks[i] for i in train_indices]
    val_msks = [all_msks[i] for i in val_indices]

    print(f"Total: {n}. Train: {len(train_imgs)}, Val: {len(val_imgs)}")
    
    # 3. Dataloaders
    train_ds = SegTileDataset(train_imgs, train_msks, img_size=IMG_SIZE)
    val_ds = SegTileDataset(val_imgs, val_msks, img_size=IMG_SIZE)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    # 4. Build Model
    base_resnet = build_encoder_resnet50()
    feat_encoder = ResNetBackboneFeatures(base_resnet)
    model = UNetFromEncoder(feat_encoder, n_classes=1).to(DEVICE)

    # 5. Load Encoder Weights (Exact Logic)
    if os.path.exists(encoder_weights_path):
        try:
            print(f"Loading encoder weights from {encoder_weights_path}...")
            sd = torch.load(encoder_weights_path, map_location=DEVICE)
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            
            base_state = base_resnet.state_dict()
            filtered = {}
            for k_sd, v_sd in sd.items():
                # Logic: remove 'backbone.' or 'encoder.' prefixes from SSL saving
                k_base = k_sd.replace("backbone.", "").replace("encoder.", "")
                if k_base in base_state and v_sd.shape == base_state[k_base].shape:
                    filtered[k_base] = v_sd
            
            base_resnet.load_state_dict(filtered, strict=False)
            print(f"Loaded {len(filtered)} matching encoder params.")
        except Exception as e:
            print(f"Failed to load encoder weights: {e}")
    else:
        print(f"Warning: Encoder weights not found at {encoder_weights_path}")

    loss_fn = BCEDiceLoss()
    best_val = 1e9
    history = {"train_loss": [], "train_dice": [], "val_loss": [], "val_dice": []}

    # === Phase 1: Freeze Encoder ===
    print("\n=== Phase 1: Training decoder (Encoder Frozen) ===")
    for p in feat_encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epochs_freeze):
        tr_loss, tr_dice = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss, val_dice = valid_epoch(model, val_loader, loss_fn, DEVICE)
        
        history["train_loss"].append(tr_loss); history["train_dice"].append(tr_dice)
        history["val_loss"].append(val_loss); history["val_dice"].append(val_dice)
        
        print(f"[Freeze] Epoch {epoch+1} tr_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f}")
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "unet_best.pth"))

    # === Phase 2: Unfreeze (Fine-tune) ===
    print("\n=== Phase 2: Fine-tuning full model ===")
    for p in feat_encoder.parameters():
        p.requires_grad = True
    
    # Reduced Learning Rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr/10)

    for epoch in range(epochs_finetune):
        tr_loss, tr_dice = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss, val_dice = valid_epoch(model, val_loader, loss_fn, DEVICE)
        
        history["train_loss"].append(tr_loss); history["train_dice"].append(tr_dice)
        history["val_loss"].append(val_loss); history["val_dice"].append(val_dice)
        
        print(f"[FT] Epoch {epoch+1} tr_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f}")
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "unet_best.pth"))

    # Save final
    torch.save(model.state_dict(), os.path.join(save_dir, "unet_final.pth"))
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(history, f)
    
    print(f"Training finished. Models saved to {save_dir}")

if __name__ == "__main__":
    # Example usage via CLI
    # You would typically use argparse here, but for now we keep it simple
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m solar_ssl.training.train_unet <data_dir> <encoder_weights> <save_dir>")
    else:
        train_unet(sys.argv[1], sys.argv[2], sys.argv[3])