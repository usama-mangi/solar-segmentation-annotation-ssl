import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

# Project imports
from ..models.simclr import SimCLRModel, NTXentLoss
from ..data.dataset import SatelliteImageDataset

# --- Specific Transform for SimCLR (from train_ddp.py) ---
# Note: This differs from the MoCo transform (no RandomResizedCrop)
class SimCLRDataTransform:
    def __init__(self, image_size=256):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __call__(self, image):
        view_1 = self.transform(image)
        view_2 = self.transform(image)
        return view_1, view_2

# --- DDP Helpers ---
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' 
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def run_training_job(rank, world_size, dataset_path):
    print(f"Starting job on GPU {rank}...")
    setup_ddp(rank, world_size)
    
    # --- 1. Configuration ---
    PROJECTION_DIM = 128
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    IMAGE_SIZE = 256
    BATCH_SIZE = 256 # Per GPU
    device = rank 
    
    if rank == 0:
        print(f"--- Using Checkpointing & Mixed-Precision (AMP) ---\n")
        print(f"Physical BATCH_SIZE per GPU: {BATCH_SIZE}")
        print(f"Total Effective Batch Size: {BATCH_SIZE * world_size}")
    
    # --- 2. Model & Loss ---
    model = SimCLRModel(projection_dim=PROJECTION_DIM) 
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    # find_unused_parameters=True required for checkpointing wrappers
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    criterion = NTXentLoss(batch_size=BATCH_SIZE, temperature=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- Initialize GradScaler for AMP ---
    scaler = torch.amp.GradScaler('cuda')

    # --- 3. Data ---
    data_transform = SimCLRDataTransform(image_size=IMAGE_SIZE)
    
    if not os.path.exists(dataset_path):
         if rank == 0: print(f"Error: Dataset directory not found at {dataset_path}")
         cleanup_ddp()
         return

    dataset = SatelliteImageDataset(image_dir=dataset_path, transform=data_transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=2, 
        pin_memory=True,
        drop_last=True, 
        sampler=sampler
    )

    # --- 4. Training Loop ---
    current_loss = 9999.0
    epochs_without_improvement = 0
    easing = 5
    
    for epoch in range(EPOCHS):
        dataloader.sampler.set_epoch(epoch)
        model.train()
        epoch_losses = []
        
        if rank == 0:
            batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        else:
            batch_iterator = dataloader

        for (views_1, views_2) in batch_iterator:
            views_1 = views_1.to(device, non_blocking=True)
            views_2 = views_2.to(device, non_blocking=True)
            
            optimizer.zero_grad() 
            
            # --- AMP: Forward pass ---
            with torch.amp.autocast('cuda'):
                z_i = model(views_1)
                z_j = model(views_2)
                loss = criterion(z_i, z_j)
            
            # --- AMP: Backward pass ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_losses.append(loss.item())

        avg_epoch_loss = np.mean(epoch_losses)
        
        if rank == 0:
            print(f"\n--- Epoch {epoch+1}/{EPOCHS} Complete ---")
            print(f"Average Epoch Loss (GPU 0): {avg_epoch_loss:.4f}")

            if (avg_epoch_loss < current_loss):
                epochs_without_improvement = 0
                encoder_weights_path = "weights/simclr_encoder_best.pth"
                os.makedirs("weights", exist_ok=True)
                torch.save(model.module.state_dict(), encoder_weights_path)
                print(f"Loss improved. Saved new weights to {encoder_weights_path}")
                current_loss = avg_epoch_loss
            else:
                epochs_without_improvement += 1
                print(f"No improvement. {epochs_without_improvement}/{easing}.")
                
                if(epochs_without_improvement >= easing):
                    print(f"Early stopping triggered.")
                    break 

    cleanup_ddp()
    print(f"Finished job on GPU {rank}.")

if __name__ == '__main__':
    # Usage: uv run python -m solar_ssl.training.train_simclr
    import sys
    
    # Hardcoded path from notebook or argument
    DATA_PATH = "data/processed" 
    
    WORLD_SIZE = torch.cuda.device_count()
    print(f"Spawning {WORLD_SIZE} processes for DDP training...")
    
    mp.spawn(
        run_training_job,
        args=(WORLD_SIZE, DATA_PATH),
        nprocs=WORLD_SIZE,
        join=True
    )