"""
Vision Transformer (ViT) model implementation from scratch.

This module implements the ViT architecture as described in the paper
"An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021).
It includes Patch Embedding, Position Embedding, Multi-Head Attention,
Transformer Encoder blocks, and a classification head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """
    Split image into patches and linearly project them.

    Args:
        image_size (int): Height/width of input image (assuming square).
        patch_size (int): Size of each patch (square).
        in_channels (int): Number of input channels (e.g., 3 for RGB).
        embed_dim (int): Dimension of the embedding space.
    """

    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Use a convolutional layer to extract patches and project
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

    def forward(self, x):
        """
        Forward pass of patch embedding.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Patch embeddings of shape (B, num_patches, embed_dim).
        """
        # x: (B, C, H, W) -> (B, embed_dim, H/patch, W/patch)
        x = self.proj(x)
        # Flatten spatial dimensions: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention module.

    Args:
        embed_dim (int): Total dimension of the embedding.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability for attention weights.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of multi-head attention.

        Args:
            x (torch.Tensor): Input sequence of shape (B, N, embed_dim).

        Returns:
            torch.Tensor: Output sequence of shape (B, N, embed_dim).
        """
        B, N, D = x.shape
        # Compute Q, K, V and reshape
        qkv = self.qkv(x)  # (B, N, 3 * D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values
        out = attn @ v  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).contiguous()  # (B, N, num_heads, head_dim)
        out = out.reshape(B, N, D)  # (B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with pre-layer normalization.

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dimension to embed_dim.
        dropout (float): Dropout probability.
        attention_dropout (float): Dropout for attention weights.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=1.0, dropout=0.0, attention_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout=attention_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass of a transformer block.

        Args:
            x (torch.Tensor): Input sequence (B, N, embed_dim).

        Returns:
            torch.Tensor: Output sequence (B, N, embed_dim).
        """
        # Pre-norm attention with residual
        x = x + self.attn(self.norm1(x))
        # Pre-norm MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image classification.

    Args:
        image_size (int): Input image size (assumed square).
        patch_size (int): Patch size.
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        mlp_ratio (float): Ratio for MLP hidden dimension.
        dropout (float): Dropout probability.
        attention_dropout (float): Dropout for attention.
    """

    def __init__(self,
                 image_size=32,
                 patch_size=4,
                 in_channels=3,
                 num_classes=10,
                 embed_dim=128,
                 num_heads=8,
                 num_layers=4,
                 mlp_ratio=1.0,
                 dropout=0.0,
                 attention_dropout=0.0):
        super().__init__()

        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Dropout for the sequence
        self.pos_drop = nn.Dropout(dropout)

        # Transformer layers
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attention_dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head (only on CLS token)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights with Xavier uniform for linear layers and
        normal for embeddings.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Initialize CLS token and position embeddings with small values
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        Forward pass of the Vision Transformer.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        B = x.shape[0]

        # Patch embedding: (B, num_patches, embed_dim)
        x = self.patch_embed(x)

        # Prepend CLS token: (B, 1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, embed_dim)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Final layer norm
        x = self.norm(x)

        # Take the CLS token representation
        cls_out = x[:, 0]  # (B, embed_dim)

        # Classification head
        logits = self.head(cls_out)  # (B, num_classes)

        return logits


# Quick test if run directly
if __name__ == "__main__":
    # Test with random input
    model = VisionTransformer(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=128,
        num_heads=8,
        num_layers=4
    )
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Expected: (4, 10)