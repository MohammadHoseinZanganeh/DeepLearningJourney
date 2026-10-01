"""
U-Net model for semantic segmentation with optional Batch Normalization.

Supports 4 configurations via config.yaml:
  1. Bilinear upsample  + skip connections ON
  2. Bilinear upsample  + skip connections OFF
  3. Transposed conv    + skip connections ON
  4. Transposed conv    + skip connections OFF

Batch Normalization can be enabled via config.yaml (use_batch_norm: true)

Architecture (3 pooling steps → 3 decoder levels):

  Input  (B, 3,   H,   W)
    │
  Enc1   (B, 64,  H,   W)  ──────────────────────────┐ skip1
    │ pool                                           │
  Enc2   (B, 128, H/2, W/2) ────────────────────┐ skip2
    │ pool                                      │
  Enc3   (B, 256, H/4, W/4) ──────────────┐ skip3
    │ pool                                │
  Bottleneck (B, 512, H/8, W/8)           │
    │ upsample                            │
  Dec3   (B, 256, H/4, W/4) ←─────────── skip3
    │ upsample                        │
  Dec2   (B, 128, H/2, W/2) ←──────── skip2
    │ upsample                   │
  Dec1   (B, 64,  H,   W)  ←─── skip1
    │
  Output (B, num_classes, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Building block: two conv3x3 + (optional BN) + ReLU

class DoubleConv(nn.Module):
    """
    Two consecutive Conv2d(3x3, same padding) + (optional BN) + ReLU layers.

    Used for both encoder and decoder blocks.
    Does NOT change spatial dimensions (padding=1).

    Args:
        in_ch:   number of input channels
        out_ch:  number of output channels
        use_bn:  if True, add BatchNorm2d after each conv
    """

    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = False):
        super().__init__()
        
        layers = []
        
        # first conv
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        
        # second conv
        layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# U-Net

class SimpleUNet(nn.Module):
    """
    U-Net for semantic segmentation with configurable upsampling, skip connections,
    and optional Batch Normalization.

    The encoder has 3 MaxPool layers, so the decoder has exactly 3 upsample
    levels to return to the original resolution.

    Upsample modes
    
    "bilinear"   : F.interpolate(scale_factor=2) followed by a 3x3 conv that
                   halves the channel count before the skip-cat.
    "transposed" : ConvTranspose2d(stride=2) that simultaneously upsamples and
                   halves channels.

    Skip connections
    
    True  : encoder feature map is concatenated → decoder input has 2× channels.
    False : a zero tensor of the same shape replaces the encoder feature map.

    Args:
        config: dict loaded from config.yaml
    """

    def __init__(self, config: dict):
        super().__init__()

        num_classes = config["classes"]["num_classes"]          # 10
        self.upsample_mode = config["model"]["upsample_mode"]   # "bilinear" or "transposed"
        self.skip_connections = config["model"]["skip_connections"]  # True or False
        use_bn = config["model"].get("use_batch_norm", False)   # read BN setting from config

        # Encoder 
        self.enc1 = DoubleConv(3,   64, use_bn=use_bn)
        self.enc2 = DoubleConv(64,  128, use_bn=use_bn)
        self.enc3 = DoubleConv(128, 256, use_bn=use_bn)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512, use_bn=use_bn)

        # Decoder upsample layers
        if self.upsample_mode == "transposed":
            self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.up1 = nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2)
        else:
            # 3x3 convs to halve channels after bilinear interpolation
            self.up3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
            self.up2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
            self.up1 = nn.Conv2d(128, 64,  kernel_size=3, padding=1)

        # Decoder conv blocks
        self.dec3 = DoubleConv(512, 256, use_bn=use_bn)
        self.dec2 = DoubleConv(256, 128, use_bn=use_bn)
        self.dec1 = DoubleConv(128, 64,  use_bn=use_bn)

        # Final 1x1 conv to class logits
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _upsample(self, x: torch.Tensor, up_layer: nn.Module) -> torch.Tensor:
        """Apply the correct upsampling depending on self.upsample_mode."""
        if self.upsample_mode == "transposed":
            return up_layer(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            return up_layer(x)

    def _merge(self, upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Concatenate upsampled feature map with encoder skip feature map."""
        if not self.skip_connections:
            skip = torch.zeros_like(skip)
        return torch.cat([upsampled, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder
        d3 = self._upsample(b, self.up3)
        d3 = self._merge(d3, e3)
        d3 = self.dec3(d3)

        d2 = self._upsample(d3, self.up2)
        d2 = self._merge(d2, e2)
        d2 = self.dec2(d2)

        d1 = self._upsample(d2, self.up1)
        d1 = self._merge(d1, e1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)


def build_model(config: dict) -> SimpleUNet:
    """Instantiate a SimpleUNet from a config dict."""
    return SimpleUNet(config)


# simple test
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    sys.path.append(str(Path(__file__).parent.parent))
    from data.data_loader import load_config
    
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = load_config(str(config_path))
    
    fake_image = torch.randn(1, 3, 256, 256)
    
    # test without BN
    print("Testing without BatchNorm")
    config["model"]["upsample_mode"] = "transposed"
    config["model"]["skip_connections"] = True
    config["model"]["use_batch_norm"] = False
    model = SimpleUNet(config)
    output = model(fake_image)
    print(f"  Input: {fake_image.shape}, Output: {output.shape}")
    
    # test with BN
    print("\nTesting with BatchNorm")
    config["model"]["use_batch_norm"] = True
    model = SimpleUNet(config)
    output = model(fake_image)
    print(f"  Input: {fake_image.shape}, Output: {output.shape}")
    
    print("\nModel works fine!")