import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

# ====== Encoder wrapper ======
def build_encoder_resnet50(pretrained=False, in_ch=3):
    """
    Exact implementation from fine-tuned-final-unet.ipynb
    """
    model = resnet50(weights=None) # Changed pretrained=False to weights=None for modern torchvision
    if in_ch != 3:
        model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

class ResNetBackboneFeatures(nn.Module):
    """
    Extracts 5 feature maps from ResNet50.
    """
    def __init__(self, base_resnet):
        super().__init__()
        b = base_resnet
        self.conv1_block = nn.Sequential(b.conv1, b.bn1, b.relu)
        self.maxpool = b.maxpool
        self.layer1 = b.layer1
        self.layer2 = b.layer2
        self.layer3 = b.layer3
        self.layer4 = b.layer4

    def forward(self, x):
        c1 = self.conv1_block(x)
        x = self.maxpool(c1)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c1, c2, c3, c4, c5]

# ====== Decoder (U-Net style) ======
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        
        # Exact padding logic from notebook to handle size mismatches
        if x.size(2) < skip.size(2) or x.size(3) < skip.size(3):
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(x, [0, diffX, 0, diffY])
        elif x.size(2) > skip.size(2) or x.size(3) > skip.size(3):
            diffY = x.size(2) - skip.size(2)
            diffX = x.size(3) - skip.size(3)
            x = x[:, :, diffY // 2 : diffY // 2 + skip.size(2), diffX // 2 : diffX // 2 + skip.size(3)]

        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNetFromEncoder(nn.Module):
    def __init__(self, encoder_feat_model, n_classes=1):
        super().__init__()
        self.encoder = encoder_feat_model
        
        self.center_conv = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024)
        )
        self.dec4 = DecoderBlock(in_ch=1024, skip_ch=1024, out_ch=512)
        self.dec3 = DecoderBlock(in_ch=512, skip_ch=512, out_ch=256)
        self.dec2 = DecoderBlock(in_ch=256, skip_ch=256, out_ch=128)
        self.dec1 = DecoderBlock(in_ch=128, skip_ch=64, out_ch=64)
        
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        skips = self.encoder(x)
        c1, c2, c3, c4, c5 = skips
        x = self.center_conv(c5)
        x = self.dec4(x, c4)
        x = self.dec3(x, c3)
        x = self.dec2(x, c2)
        x = self.dec1(x, c1)
        x = self.dec0(x)
        logits = self.final(x)
        return logits