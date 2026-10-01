"""
model.py
All model architectures for the homework.
Includes Block A (Residual), Block B (Inception), and Block C (ResNeXt-style).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Block A: Residual Blocks
# ============================================================

class ResidualBlockType1(nn.Module):
    """
    Block A Type 1: Same channels, identity shortcut, ReLU after addition.
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockType1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu_out(out + identity)
        return out


class ResidualBlockType2(nn.Module):
    """
    Block A Type 2: Different channels, 1x1 conv shortcut (no BN), ReLU after addition.
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockType2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu_out(out + identity)
        return out


# ============================================================
# Block B: Inception-style Block
# ============================================================

class InceptionBlockB(nn.Module):
    """
    Block B: 4 parallel branches, outputs concatenated.
    
    Branch 1: 1x1 Conv -> 3x3 Conv -> 3x3 Conv
    Branch 2: 1x1 Conv -> 3x3 Conv
    Branch 3: 1x1 Conv
    Branch 4: AvgPool -> 1x1 Conv
    """
    def __init__(self, in_channels, out_channels):
        super(InceptionBlockB, self).__init__()
        branch_out = out_channels // 4
        mid_channels = max(branch_out * 2 // 3, 16)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, branch_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(branch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_out, branch_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(branch_out),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, branch_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(branch_out),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_out, 1, bias=False),
            nn.BatchNorm2d(branch_out),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            nn.Conv2d(in_channels, branch_out, 1, bias=False),
            nn.BatchNorm2d(branch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)


# ============================================================
# Block C: ResNeXt-style Block
# ============================================================

class ResNeXtBlockC(nn.Module):
    """
    Block C: ResNeXt-style block combining residual and parallel branch ideas.
    
    Structure exactly as diagram:
        Input (c channels)
            |
            ├──→ 1x1 Conv(c/g) → 3x3 Conv(c/g) ──┐
            ├──→ 1x1 Conv(c/g) → 3x3 Conv(c/g) ──┤
            ├──→ ... (g groups) ...              │──→ Concat → 1x1 Conv → + → Output
            │                                                              │
            ─────────────────Shortcut ─────────────────────────────────────┘
    
    Args:
        in_channels: Number of input channels (c).
        out_channels: Number of output channels (c).
        groups: Number of parallel branches (g).
    """

    def __init__(self, in_channels, out_channels, groups=4):
        super(ResNeXtBlockC, self).__init__()

        self.groups = groups
        c_per_group = out_channels // groups

        # Create g parallel branches: 1x1 Conv -> 3x3 Conv
        self.branches = nn.ModuleList()
        for _ in range(groups):
            branch = nn.Sequential(
                nn.Conv2d(in_channels, c_per_group, kernel_size=1, bias=False),
                nn.Conv2d(c_per_group, c_per_group, kernel_size=3, stride=1,
                         padding=1, bias=False)
            )
            self.branches.append(branch)

        # 1x1 Conv after concatenation
        self.proj = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

        # Shortcut
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        # Run all parallel branches
        branch_outputs = [branch(x) for branch in self.branches]

        # Concatenate along channel dimension
        out = torch.cat(branch_outputs, dim=1)

        # 1x1 projection
        out = self.proj(out)

        # Residual connection after concat and 1x1 conv
        identity = self.shortcut(x)
        out = out + identity

        return out


# ============================================================
# Model Factory: Creates model based on block type
# ============================================================

class BaseModel(nn.Module):
    """
    Unified architecture following Table 1.
    Supports Block A, Block B, and Block C via block_type parameter.
    
    Table 1:
    | Step | Layer              | In  | Out | BN/Pooling   |
    |------|--------------------|-----|-----|--------------|
    | 1    | Conv 3x3           | 1   | 32  | BN           |
    |      | ReLU               | -   | -   | -            |
    | 2    | Conv 3x3           | 32  | 64  | BN           |
    |      | ReLU               | -   | -   | MaxPool      |
    | 3    | Block_X            | 64  | 64  | -            |
    | 4    | Block_X            | 64  | 128 | MaxPool      |
    | 5    | Conv 3x3           | 128 | 256 | BN           |
    |      | ReLU               | -   | -   | MaxPool      |
    | 6    | Block_X            | 256 | 256 | AveragePool  |
    | 7    | Linear             | 256 | 10  | -            |
    | 8    | Softmax            | 10  | 10  | -            |
    """

    def __init__(self, block_type='A', num_classes=10, dropout_rate=0.5, **block_kwargs):
        """
        Args:
            block_type: 'A' for Residual, 'B' for Inception, 'C' for ResNeXt.
            num_classes: Number of output classes.
            dropout_rate: Dropout probability before classifier.
            block_kwargs: Extra arguments passed to the block.
                          For Block C: groups (int), bottleneck_channels (int or None).
        """
        super(BaseModel, self).__init__()
        
        # Select block class based on type
        if block_type == 'A':
            block_same = lambda in_ch, out_ch: ResidualBlockType1(in_ch, out_ch)
            block_diff = lambda in_ch, out_ch: ResidualBlockType2(in_ch, out_ch)
        elif block_type == 'B':
            block_same = lambda in_ch, out_ch: InceptionBlockB(in_ch, out_ch)
            block_diff = lambda in_ch, out_ch: InceptionBlockB(in_ch, out_ch)
        elif block_type == 'C':
            groups = block_kwargs.get('groups', 4)
            block_same = lambda in_ch, out_ch: ResNeXtBlockC(in_ch, out_ch, groups=groups)
            block_diff = lambda in_ch, out_ch: ResNeXtBlockC(in_ch, out_ch, groups=groups)
        else:
            raise ValueError(f"Unknown block_type: {block_type}. Use 'A', 'B', or 'C'.")

        # Step 1: Conv3x3(1->32) + BN + ReLU
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        # Step 2: Conv3x3(32->64) + BN + ReLU + MaxPool
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Step 3: Block (64->64) - same channels
        self.block3 = block_same(64, 64)

        # Step 4: Block (64->128) - different channels + MaxPool
        self.block4 = block_diff(64, 128)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Step 5: Conv3x3(128->256) + BN + ReLU + MaxPool
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, 2)

        # Step 6: Block (256->256) + AveragePool
        self.block6 = block_same(256, 256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #to concat to a fully connected

        # Step 7: Dropout + Linear(256->10)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Step 1
        x = self.relu1(self.bn1(self.conv1(x)))              # 1x28x28 -> 32x28x28
        # Step 2
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))  # -> 64x14x14
        # Step 3
        x = self.block3(x)                                    # -> 64x14x14
        # Step 4
        x = self.pool4(self.block4(x))                        # -> 128x7x7
        # Step 5
        x = self.pool5(self.relu3(self.bn3(self.conv3(x))))  # -> 256x3x3
        # Step 6
        x = self.avgpool(self.block6(x))                      # -> 256x1x1
        # Step 7
        x = torch.flatten(x, 1)
        x = self.fc(self.dropout(x))                          # -> 10
        return x

    def predict(self, x):
        return self.forward(x)


class TransferModel(nn.Module):
    """
    Transfer learning model for Fashion MNIST.

    Takes a pretrained BaseModel (trained on MNIST digits), freezes all
    convolutional layers as the feature extractor, and replaces only the
    final classifier layer. Only the new classifier is trained on Fashion MNIST.

    This is an example of feature extraction transfer learning: the
    frozen conv layers apply the same edge, texture, and shape detection
    learned from digits to clothing images.
    """

    def __init__(self, pretrained_model, num_classes=10, dropout_rate=0.5):
        """
        Args:
            pretrained_model: A trained BaseModel instance.
            num_classes: Number of Fashion MNIST classes (10).
            dropout_rate: Dropout probability for the new classifier.
        """
        super(TransferModel, self).__init__()

        # Copy all convolutional layers from the pretrained model.
        self.conv1 = pretrained_model.conv1
        self.bn1 = pretrained_model.bn1
        self.relu1 = pretrained_model.relu1

        self.conv2 = pretrained_model.conv2
        self.bn2 = pretrained_model.bn2
        self.relu2 = pretrained_model.relu2
        self.pool2 = pretrained_model.pool2

        self.block3 = pretrained_model.block3

        self.block4 = pretrained_model.block4
        self.pool4 = pretrained_model.pool4

        self.conv3 = pretrained_model.conv3
        self.bn3 = pretrained_model.bn3
        self.relu3 = pretrained_model.relu3
        self.pool5 = pretrained_model.pool5

        self.block6 = pretrained_model.block6
        self.avgpool = pretrained_model.avgpool

        # Freeze all feature extractor layers.
        for param in self.parameters():
            param.requires_grad = False

        # New trainable classifier head.
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        """Forward pass: frozen feature extraction + new classifier."""
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.block3(x)
        x = self.pool4(self.block4(x))
        x = self.pool5(self.relu3(self.bn3(self.conv3(x))))
        x = self.avgpool(self.block6(x))
        x = torch.flatten(x, 1)
        x = self.fc(self.dropout(x))
        return x

# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Testing all model architectures...")
    print("=" * 50)

    x = torch.randn(4, 1, 28, 28)

    for block_type in ['A', 'B', 'C']:
        model = BaseModel(block_type=block_type, num_classes=10, dropout_rate=0.0)
        total_params = sum(p.numel() for p in model.parameters())
        output = model(x)
        print(f"\nBlock {block_type}:")
        print(f"  Parameters: {total_params:,}")
        print(f"  Input:  {x.shape}")
        print(f"  Output: {output.shape}")
    
    print("\nAll models tested successfully.")