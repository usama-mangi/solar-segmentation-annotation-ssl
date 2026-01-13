import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

from ..models.moco import MoCo, EncoderProjector
from ..data.dataset import SatelliteImageDataset
from ..data.transforms import SimCLRDataTransform

def train_moco(data_dir, epochs=100, batch_size=64, lr=1e-4, device='cuda'):
    # Hyperparams from Cell 9
    NUM_EPOCHS = epochs
    PATIENCE = 5
    PROJECTION_DIM = 128
    QUEUE_SIZE = 4096
    MOMENTUM = 0.999
    TEMPERATURE = 0.07
    
    # Data Setup
    print("Setting up data...")
    transform = SimCLRDataTransform(image_size=256) # Image Size from Cell 3
    dataset = SatelliteImageDataset(data_dir, transform)
    
    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    
    # Model Setup
    model = MoCo(
        base_encoder_cls=EncoderProjector,
        dim=PROJECTION_DIM,
        K=QUEUE_SIZE,
        m=MOMENTUM,
        T=TEMPERATURE
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.encoder_q.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()

    # Training Loop State
    epoch_no_improve = 0
    best_val_loss = np.inf
    best_model_path = "weights/moco_encoder_best.pth"
    
    print(f"Starting training on {device}...")

    try:
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} (Train)")
            for (views_1, views_2) in pbar:
                im_q = views_1.to(device)
                im_k = views_2.to(device)
                
                with torch.cuda.amp.autocast():
                    logits, labels = model(im_q, im_k)
                    loss = criterion(logits, labels)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (views_1, views_2) in tqdm(val_loader, desc=f"Epoch {epoch+1} (Val)"):
                    im_q = views_1.to(device)
                    im_k = views_2.to(device)
                    
                    with torch.cuda.amp.autocast():
                        logits, labels = model(im_q, im_k)
                        loss = criterion(logits, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch+1}/{NUM_EPOCHS} Complete.")
            print(f"Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                print(f"Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
                best_val_loss = avg_val_loss
                epoch_no_improve = 0
                torch.save(model.encoder_q.encoder.state_dict(), best_model_path)
            else:
                epoch_no_improve += 1
                print(f"No improvement for {epoch_no_improve} epochs.")
            
            if epoch_no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break
                
    except Exception as e:
        print(f"Error during training: {e}")