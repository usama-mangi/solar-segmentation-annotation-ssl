import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torch.utils.checkpoint import checkpoint

class CheckpointWrapper(nn.Module):
    """
    Wrapper to enable activation checkpointing with DDP compatibility.
    Ref: encoder-simclr.ipynb (train_ddp.py)
    """
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, x):
        # use_reentrant=False is required for SyncBatchNorm compatibility
        return checkpoint(self.module, x, use_reentrant=False)

class SimCLRModel(nn.Module):
    """
    SimCLR Encoder with ResNet50 backbone and Projection Head.
    Includes explicit activation checkpointing for layers 1-4.
    """
    def __init__(self, encoder_output_dim=2048, projection_dim=128):
        super(SimCLRModel, self).__init__()
        
        # 1. Encoder (ResNet-50)
        self.encoder = resnet50(weights='DEFAULT')
        
        # Remove classification layer
        num_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity() 
        
        # 2. Projection Head
        self.projector = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, projection_dim)
        )
        
        # --- ENABLE CHECKPOINTING ---
        # Logic from train_ddp.py Cell 2
        self.encoder.layer1 = CheckpointWrapper(self.encoder.layer1)
        self.encoder.layer2 = CheckpointWrapper(self.encoder.layer2)
        self.encoder.layer3 = CheckpointWrapper(self.encoder.layer3)
        self.encoder.layer4 = CheckpointWrapper(self.encoder.layer4)

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projector(features)
        return projections

class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.mask = self.create_positive_mask(batch_size)

    def create_positive_mask(self, batch_size):
        N = 2 * batch_size
        mask = torch.zeros((N, N), dtype=torch.bool)
        for i in range(batch_size):
            mask[i, batch_size + i] = 1
            mask[batch_size + i, i] = 1
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        device = z_i.device 
        
        z = torch.cat((z_i, z_j), dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim = sim / self.temperature
        
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask_on_device = self.mask.to(device)
        
        negative_mask = ~mask_on_device
        negative_samples = sim[negative_mask].reshape(N, -1)
        
        labels = torch.zeros(N).to(device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        
        loss = self.criterion(logits, labels)
        loss /= N
        return loss