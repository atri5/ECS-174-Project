import torch
import torch.nn as nn
import math

# Patch embedding for ViT
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # Shape: (batch_size, embed_dim, num_patches_sqrt, num_patches_sqrt)
        x = x.flatten(2)  # Shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)
        return x

# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

    def forward(self, x):
        return x + self.pos_embedding

# Multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        qkv = self.qkv_proj(x)  # Shape: (batch_size, seq_len, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Shape: (3, batch_size, num_heads, seq_len, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape: (batch_size, num_heads, seq_len, head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # Shape: (batch_size, num_heads, seq_len, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # Shape: (batch_size, num_heads, seq_len, head_dim)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        attn_output = self.o_proj(attn_output)
        attn_output = self.dropout(attn_output)
        return attn_output

# Feedforward network
class FeedForward(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.0):
        super(FeedForward, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)

# Transformer encoder layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # Multi-head attention
        x = x + self.ffn(self.norm2(x))  # Feed-forward network
        return x

# Vision Transformer with dual output: classification and regression
class VisionTransformerWithCoordinates(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1,
                 num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_dim=3072, dropout=0.0):
        super(VisionTransformerWithCoordinates, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Positional encoding
        self.pos_embed = PositionalEncoding(num_patches, embed_dim)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.cls_head = nn.Linear(embed_dim, num_classes)  # Classification head
        self.coord_head = nn.Linear(embed_dim, 2)  # Coordinates regression head (2 values for x, y)

    def forward(self, x):
        x = self.patch_embed(x)  # Shape: (batch_size, num_patches, embed_dim)

        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: (batch_size, num_patches + 1, embed_dim)

        x = self.pos_embed(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)
        cls_output = x[:, 0]  # Extract the CLS token output for classification

        # Predict severity (classification)
        severity_logits = self.cls_head(cls_output)

        # Predict coordinates (regression)
        coords_output = self.coord_head(cls_output)  # Predict (x, y) coordinates

        return severity_logits, coords_output
