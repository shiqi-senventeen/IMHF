# Molecular prediction model based on meta-learning
import math
import os

import torch
import torch.nn as nn

from einops import rearrange
from config import args
import numpy as np
import matplotlib.pyplot as plt


class CNNTransformer(nn.Module):
    """
    CNN-Transformer hybrid model for molecular prediction.
    Combines CNN-based feature extraction with Transformer-based attention mechanism.
    """

    def __init__(self, n_channels, imgsize, patch_num=28, dim=1, depth=12, heads=4, mlp_dim=512 * 4, dim_head=64,
                 dropout=0.1, emb_dropout=0.1):
        """
        Initialize the CNN-Transformer model.

        Args:
            n_channels (int): Number of input image channels
            imgsize (int): Size of input image (assumed square)
            patch_num (int): Number of patches along one dimension
            dim (int): Embedding dimension
            depth (int): Number of transformer layers
            heads (int): Number of attention heads
            mlp_dim (int): Dimension of feed-forward network
            dim_head (int): Dimension of each attention head
            dropout (float): Dropout rate
            emb_dropout (float): Embedding dropout rate
        """
        super().__init__()
        # Set image dimensions
        self.image_height, self.image_width = imgsize, imgsize
        # Set patch dimensions
        self.patch_height, self.patch_width = 224, 224
        self.dmodel = dim

        # Ensure patches divide image evenly
        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = patch_num * patch_num

        # CNN encoder for feature extraction
        self.cnn_encoder = CNNEncoder2(n_channels, dim, self.patch_height, self.patch_width)  # the original is CNNs

        # Transformer for attention mechanism
        self.transformer = CNNTransformer_record(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches)

        # Linear layers for latent space transformations
        self.linear1 = nn.Linear(in_features=784, out_features=768)
        self.linear2 = nn.Linear(in_features=768, out_features=784)

        # Decoder network to reconstruct images from latent features
        self.decoder = nn.Sequential(
            nn.Conv2d(self.dmodel, 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(3, 3, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(inplace=True)
        )
        # Loss function
        self.criterion = nn.MSELoss()

    def forward(self, img):
        """
        Forward pass through the model.

        Args:
            img (torch.Tensor): Input image tensor of shape [B, C, H, W]

        Returns:
            x (torch.Tensor): Reconstructed image
            latent (torch.Tensor): Latent representation
            loss (torch.Tensor): Reconstruction loss
        """
        B, c, h, w = img.shape

        img = img.view(B, c, h, w)
        # Pass through CNN encoder
        x = self.cnn_encoder(img)

        # Pass through transformer
        x = self.transformer(x)  # b c h w -> b c h w

        # Flatten spatial dimensions
        x = torch.flatten(x, start_dim=-2, end_dim=-1)  # [b,c,h,w]->[b,c,h*w]
        # Project to latent space
        latent = self.linear1(x)

        # Project back to original dimensions
        x = self.linear2(latent)

        # Reshape for decoder
        x = x.reshape(-1, 1, 28, 28)
        # Decode to reconstruct image
        x = self.decoder(x)

        setsz = latent.size(0)
        latent = latent.view(setsz, -1)
        # Calculate reconstruction loss
        loss = self.criterion(x, img)

        return x, latent, loss

    def infere(self, img):
        """
        Inference method with attention map recording.

        Args:
            img (torch.Tensor): Input image tensor

        Returns:
            x (torch.Tensor): Reconstructed image
            ftokens (list): Feature tokens at each layer
            attmaps (list): Attention maps at each layer
        """
        # Encode image
        x0 = self.cnn_encoder(img)
        # Pass through transformer with attention recording
        x, ftokens, attmaps = self.transformer.infere(x0)
        # Add initial tokens
        ftokens.insert(0, rearrange(x0, 'b c h w -> b (h w) c'))
        # Decode
        x = self.decoder(x)
        return x, ftokens, attmaps


class CNNEncoder2(nn.Module):
    """
    CNN encoder for feature extraction.
    Uses a series of convolutional blocks with downsampling.
    """

    def __init__(self, n_channels, out_channels, patch_height, patch_width):
        """
        Initialize the CNN encoder.

        Args:
            n_channels (int): Number of input channels
            out_channels (int): Number of output channels
            patch_height (int): Height of each patch
            patch_width (int): Width of each patch
        """
        super(CNNEncoder2, self).__init__()
        self.scale = 1
        # Convolution-BatchNorm-ReLU blocks
        self.inc = SingleConv(n_channels, 64 // self.scale)
        self.down1 = DownSingleConv(64 // self.scale, 128 // self.scale)
        self.down2 = DownSingleConv(128 // self.scale, 256 // self.scale)
        self.down3 = DownSingleConv(256 // self.scale, out_channels)

    def forward(self, x):
        """
        Forward pass through CNN encoder.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Encoded features
        """
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x


class DownSingleConv(nn.Module):
    """
    Downscaling block with pooling followed by convolution.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize the downsampling block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            SingleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """
        Forward pass through downsampling block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Downsampled features
        """
        return self.maxpool_conv(x)


class SingleConv(nn.Module):
    """
    Single convolutional block with BatchNorm and ReLU.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize the convolutional block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super().__init__()
        self.CBR = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through conv block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Processed features
        """
        return self.CBR(x)


class CNNTransformer_record(nn.Module):
    """
    Transformer module adapted for CNN feature maps.
    Records attention maps during inference.
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=784):
        """
        Initialize the CNN Transformer.

        Args:
            dim (int): Model dimension
            depth (int): Number of transformer layers
            heads (int): Number of attention heads
            dim_head (int): Dimension of each attention head
            mlp_dim (int): Dimension of feed-forward network
            dropout (float): Dropout rate
            num_patches (int): Number of patches (spatial positions)
        """
        super().__init__()
        # Create transformer layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CNNAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches),
                CNNFeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        """
        Standard forward pass through transformer layers.

        Args:
            x (torch.Tensor): Input feature map tensor [B, C, H, W]

        Returns:
            x (torch.Tensor): Processed feature map
        """
        for attn, ff in self.layers:
            x = attn(x) + x  # Attention block with residual connection
            x = ff(x) + x  # Feed-forward block with residual connection
        return x

    def infere(self, x):
        """
        Inference with recording of attention maps and feature tokens.

        Args:
            x (torch.Tensor): Input feature map tensor

        Returns:
            x (torch.Tensor): Processed feature map
            ftokens (list): Feature tokens at each layer
            attmaps (list): Attention maps at each layer
        """
        ftokens, attmaps = [], []
        for attn, ff in self.layers:
            ax, amap = attn(x, mode="record")
            x = ax + x
            x = ff(x) + x
            ftokens.append(rearrange(x, 'b c h w -> b (h w) c'))
            attmaps.append(amap)
        return x, ftokens, attmaps


class CNNAttention(nn.Module):
    """
    Attention mechanism adapted for CNN feature maps.
    Incorporates relative position encoding.
    """

    def __init__(self, dim, heads=4, dim_head=64, dropout=0., num_patches=784):
        """
        Initialize the CNN Attention module.

        Args:
            dim (int): Input dimension
            heads (int): Number of attention heads
            dim_head (int): Dimension of each attention head
            dropout (float): Dropout rate
            num_patches (int): Number of spatial positions
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.num_patches = num_patches

        # QKV projection with 3x3 convolution
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=3, padding=1, bias=False)

        # Relative position encoding
        self.dis = relative_pos_dis(math.sqrt(num_patches), math.sqrt(num_patches), sita=0.9).to(args.device)
        self.headsita = nn.Parameter(torch.randn(heads), requires_grad=True)
        self.sig = nn.Sigmoid()

        # Output projection
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        ) if project_out else nn.Identity()

    def forward(self, x, mode="train", smooth=1e-4):
        """
        Forward pass through attention layer.

        Args:
            x (torch.Tensor): Input feature map [B, C, H, W]
            mode (str): "train" for standard forward, "record" for recording attention maps
            smooth (float): Small epsilon for numerical stability

        Returns:
            torch.Tensor or tuple: Processed feature map, and optionally attention maps
        """
        # Project to queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (g d) h w -> b g (h w) d', g=self.heads), qkv)

        # Calculate attention scores
        attn = torch.matmul(q, k.transpose(-1, -2))  # b g n n

        # Normalize attention scores by feature magnitude
        qk_norm = torch.sqrt(torch.sum(q ** 2, dim=-1) + smooth)[:, :, :, None] * torch.sqrt(
            torch.sum(k ** 2, dim=-1) + smooth)[:, :, None, :] + smooth
        attn = attn / qk_norm

        # Apply position-based attention modulation
        factor = 1 / (2 * (self.sig(self.headsita) + 0.01) ** 2)  # h
        factor = 1 / (2 * (self.sig(self.headsita) * (
                0.4 - 0.003) + 0.003) ** 2)  # af3 + limited setting this, or using the above line code
        dis = factor[:, None, None] * self.dis[None, :, :]  # g n n
        dis = torch.exp(-dis)
        dis = dis / torch.sum(dis, dim=-1)[:, :, None]

        # Modulate attention with positional weights
        attn = attn * dis[None, :, :, :]

        sample = attn[0, 1]

        if isinstance(sample, torch.Tensor):
            sample = sample.detach().cpu().numpy()

        plt.show()
        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape back to feature map format
        out = rearrange(out, 'b g (h w) d -> b (g d) h w', h=x.shape[2])

        if mode == "train":
            return self.to_out(out)
        else:
            return self.to_out(out), attn


class CNNFeedForward(nn.Module):
    """
    Feed-forward network for transformer layers.
    Uses 1x1 convolutions for position-wise operations.
    """

    def __init__(self, dim, hidden_dim, dropout=0.):
        """
        Initialize the feed-forward network.

        Args:
            dim (int): Input dimension
            hidden_dim (int): Hidden dimension
            dropout (float): Dropout rate
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through feed-forward network.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Processed tensor
        """
        return self.net(x)


def relative_pos_dis(height=32, weight=32, sita=0.9):
    """
    Calculate relative position distances matrix.
    Used for position-based attention modulation.

    Args:
        height (int): Grid height
        weight (int): Grid width
        sita (float): Position encoding parameter

    Returns:
        torch.Tensor: Distance matrix between all positions
    """
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww # 0 is 32 * 32 for h, 1 is 32 * 32 for w
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    dis = (relative_coords[:, :, 0].float() / height) ** 2 + (relative_coords[:, :, 1].float() / weight) ** 2
    return dis


if __name__ == '__main__':
    # Test the model with random input
    tensor = torch.randn(6, 3, 224, 224).to("cuda")

    print(tensor.size())

    setr = CNNTransformer(3, 224)
    setr.to("cuda")
    x, latent = setr(tensor)
    print(x.shape, latent.size())