import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class EncoderProjector(nn.Module):
    """
    1. Encoder (Backbone): ResNet-50.
    2. Projection Head: MLP.
    """
    def __init__(self, encoder_output_dim=2048, projection_dim=128):
        super(EncoderProjector, self).__init__()
        
        # 1. Encoder (ResNet-50)
        self.encoder = resnet50(weights='DEFAULT')
        num_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity() # Aakhri layer hata di
        
        # 2. Projection Head (g)
        self.projector = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, projection_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projector(features)
        return projections

class MoCo(nn.Module):
    """
    MoCo (Momentum Contrast) implementation.
    """
    def __init__(self, base_encoder_cls=EncoderProjector, dim=128, K=4096, m=0.999, T=0.07):
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # Do encoders banayein: query aur key
        self.encoder_q = base_encoder_cls(projection_dim=dim)
        self.encoder_k = base_encoder_cls(projection_dim=dim)

        # Key encoder parameters copy and freeze
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Negative samples queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        # 1. Query features
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        # 2. Key features
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)

        # 3. Positive logits
        l_pos = (q * k).sum(dim=1).unsqueeze(-1)

        # 4. Negative logits
        l_neg = torch.matmul(q, self.queue.clone().detach())

        # 5. Combine
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        self._dequeue_and_enqueue(k)

        return logits, labels