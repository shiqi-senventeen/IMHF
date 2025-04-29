# -*-coding:utf-8-*-
# Molecular prediction model based on meta-learning
import math
import os

import torch
import torch.nn as nn

from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torchvision.transforms as transforms
# 元模型
from rdkit import Chem
from rdkit.Chem import Draw, AllChem

from Module.datautils import args


# class CNNTransformer(nn.Module):
#     def __init__(self, n_channels, imgsize, patch_num=28, dim=1, depth=12, heads=4, mlp_dim=512*4, dim_head=64, dropout=0.1, emb_dropout=0.1):
#         super().__init__()
#         # H,W=224,224
#         self.image_height, self.image_width = imgsize,imgsize
#         # pacth的高宽(8,8)
#         self.patch_height, self.patch_width = 224,224
#         self.dmodel = dim
#
#         # 检查patch_height能否被image_height整除
#         assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0,\
#             'Image dimensions must be divisible by the patch size.'
#
#         num_patches = patch_num * patch_num
#
#         self.cnn_encoder = CNNEncoder2(n_channels, dim, self.patch_height, self.patch_width) # the original is CNNs
#
#         self.transformer = CNNTransformer_record(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches)
#
#         self.linear1 = nn.Linear(in_features=784,out_features=768)
#         self.linear2 = nn.Linear(in_features=768,out_features=784)
#
#         self.decoder = nn.Sequential(
#             nn.Conv2d(self.dmodel, 3, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(3),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(3),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(3, 3, kernel_size=1),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#         )
#         self.criterion = nn.MSELoss()
#
#     def forward(self, img):
#         B,c,h,w=img.shape
#
#         img=img.view(B,c,h,w)
#         x = self.cnn_encoder(img)
#         # encoder   x:[6,1,28,28]
#         '''
#         plt一下注意力图
#         '''
#         # sample =x[0]
#         # sample = sample.detach()
#         #
#         # sample_np=sample.cpu().numpy()
#         # sample=np.transpose(sample_np,(1,2,0))
#         # plt.imshow(sample, cmap='gray')
#         # plt.show()
#
#         x = self.transformer(x)  # b c h w -> b c h w
#
#         # sample = x[0]
#         # sample = sample.detach()
#         #
#         # sample_np = sample.cpu().numpy()
#         # sample = np.transpose(sample_np, (1, 2, 0))
#         # plt.imshow(sample, cmap='gray')
#         # plt.show()
#         #
#
#         '''
#         加入一个latent层
#         '''
#         x = torch.flatten(x, start_dim=-2, end_dim=-1)      # [b,c,h,w]->[b,c,h*w]
#         latent = self.linear1(x)
#
#         x =self.linear2(latent)
#
#         x = x.reshape(-1,1,28,28)
#         x = self.decoder(x)
#
#         # x:[6,3,224,224]
#         # latent:[6,1,768]
#         setsz=latent.size(0)
#         latent = latent.view(setsz, -1)
#         # latent:[6,768]
#         loss=self.criterion(x,img)
#         # 出来的张量为784*784
#         return loss,latent
#
#     def infere(self, img):
#         x0 = self.cnn_encoder(img)
#         # encoder
#         x, ftokens, attmaps = self.transformer.infere(x0)
#         ftokens.insert(0, rearrange(x0, 'b c h w -> b (h w) c'))
#         # decoder
#         x = self.decoder(x)
#         return x, ftokens, attmaps
#
#
# class CNNEncoder2(nn.Module):
#     def __init__(self, n_channels, out_channels, patch_height, patch_width):
#         super(CNNEncoder2, self).__init__()
#         self.scale = 1
#         # CBR
#         self.inc = SingleConv(n_channels, 64 // self.scale)
#         self.down1 = DownSingleConv(64 // self.scale, 128 // self.scale)
#         self.down2 = DownSingleConv(128 // self.scale, 256 // self.scale)
#         self.down3 = DownSingleConv(256 // self.scale, out_channels)
#
#     def forward(self, x):
#         x = self.inc(x)
#         x = self.down1(x)
#         x = self.down2(x)
#         x = self.down3(x)
#         return x
#
# class DownSingleConv(nn.Module):
#     """Downscaling with maxpool then double conv"""
#
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             SingleConv(in_channels, out_channels)
#         )
#
#     def forward(self, x):
#         return self.maxpool_conv(x)
#
# class SingleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.CBR = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#     def forward(self, x):
#         return self.CBR(x)
#
#
# class CNNTransformer_record(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=784):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 CNNAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches),
#                 CNNFeedForward(dim, mlp_dim, dropout=dropout)
#             ]))
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x
#     def infere(self, x):
#         ftokens, attmaps = [], []
#         for attn, ff in self.layers:
#             ax, amap = attn(x, mode="record")
#             x = ax + x
#             x = ff(x) + x
#             ftokens.append(rearrange(x, 'b c h w -> b (h w) c'))
#             attmaps.append(amap)
#         return x, ftokens, attmaps
#
#
# class CNNAttention(nn.Module):
#     def __init__(self, dim, heads=4, dim_head=64, dropout=0., num_patches=784):
#         super().__init__()
#         inner_dim = dim_head * heads
#         project_out = not (heads == 1 and dim_head == dim)
#
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         self.num_patches = num_patches
#
#         #self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=1, padding=0, bias=False)
#         self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=3, padding=1, bias=False)
#         self.dis = relative_pos_dis(math.sqrt(num_patches), math.sqrt(num_patches), sita=0.9).to(args.device)
#         self.headsita = nn.Parameter(torch.randn(heads), requires_grad=True)
#         self.sig = nn.Sigmoid()
#
#         self.to_out = nn.Sequential(
#             nn.Conv2d(inner_dim, dim, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(dim), # inner_dim
#             nn.ReLU(inplace=True),
#         ) if project_out else nn.Identity()
#
#     def forward(self, x, mode="train", smooth=1e-4):
#         qkv = self.to_qkv(x).chunk(3, dim=1)
#         q, k, v = map(lambda t: rearrange(t, 'b (g d) h w -> b g (h w) d', g=self.heads), qkv)
#         attn = torch.matmul(q, k.transpose(-1, -2)) # b g n n
#         qk_norm = torch.sqrt(torch.sum(q ** 2, dim=-1)+smooth)[:, :, :, None] * torch.sqrt(torch.sum(k ** 2, dim=-1)+smooth)[:, :, None, :] + smooth
#         attn = attn/qk_norm
#         # 记得注释掉下面一行
#         # attentionheatmap_visual2(attn, self.sig(self.headsita), out_dir='./Visualization', value=1)
#         factor = 1/(2*(self.sig(self.headsita)+0.01)**2) # h
#         factor = 1/(2*(self.sig(self.headsita)*(0.4-0.003)+0.003)**2) # af3 + limited setting this, or using the above line code
#         dis = factor[:, None, None]*self.dis[None, :, :] # g n n
#         dis = torch.exp(-dis)
#         dis = dis/torch.sum(dis, dim=-1)[:, :, None]
#         # 记得注释掉
#         # attentionheatmap_visual2(dis[None, :, :, :], self.sig(self.headsita), out_dir='./Visualization',
#         #                          value=0.003)
#         attn = attn * dis[None, :, :, :]
#         # 记得注释掉
#         # attentionheatmap_visual2(attn, self.sig(self.headsita), out_dir='./Visualization', value=0.003)
#
#         #attentionheatmap_visual(attn, out_dir='./Visualization/attention_af3/')
#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b g (h w) d -> b (g d) h w', h=x.shape[2])
#         if mode=="train":
#             return self.to_out(out)
#         else:
#             return self.to_out(out), attn
#
# class CNNFeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(dim, hidden_dim, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim, dim, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(inplace=True)
#         )
#     def forward(self, x):
#         return self.net(x)
#
#
# def relative_pos_dis(height=32, weight=32, sita=0.9):
#     coords_h = torch.arange(height)
#     coords_w = torch.arange(weight)
#     coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww # 0 is 32 * 32 for h, 1 is 32 * 32 for w
#     coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#     relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#     relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#     dis = (relative_coords[:, :, 0].float()/height) ** 2 + (relative_coords[:, :, 1].float()/weight) ** 2
#     #dis = torch.exp(-dis*(1/(2*sita**2)))
#     return  dis
class CNNTransformer(nn.Module):
    def __init__(self, n_channels, imgsize, patch_num=28, dim=1, depth=12, heads=4, mlp_dim=512*4, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        # H,W=224,224
        self.image_height, self.image_width = imgsize,imgsize
        # pacth的高宽(8,8)
        self.patch_height, self.patch_width = 224,224
        self.dmodel = dim

        # 检查patch_height能否被image_height整除
        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0,\
            'Image dimensions must be divisible by the patch size.'

        num_patches = patch_num * patch_num

        self.cnn_encoder = CNNEncoder2(n_channels, dim, self.patch_height, self.patch_width) # the original is CNNs

        self.transformer = CNNTransformer_record(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches)
# 5.23
        # self.linear1 = nn.Linear(in_features=784,out_features=768)
        # self.linear2 = nn.Linear(in_features=768,out_features=784)
        self.linear1 = nn.Linear(in_features=784,out_features=768)
        self.linear2 = nn.Linear(in_features=768,out_features=784)

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
        )
        self.criterion = nn.MSELoss()

    def forward(self, img):
        B,c,h,w=img.shape

        img=img.view(B,c,h,w)
        x = self.cnn_encoder(img)

        # encoder   x:[6,1,28,28]
        # '''
        # plt一下注意力图
        # '''
        # sample =x[0]
        # sample = sample.detach()
        #
        # sample_np=sample.cpu().numpy()
        # sample=np.transpose(sample_np,(1,2,0))
        # plt.imshow(sample, cmap='gray')
        # plt.show()

        x = self.transformer(x)  # b c h w -> b c h w

        # sample = x[0]
        # sample = sample.detach()
        #
        # sample_np = sample.cpu().numpy()
        # sample = np.transpose(sample_np, (1, 2, 0))
        # plt.imshow(sample, cmap='gray')
        # plt.show()
        #

        '''
        加入一个latent层
        '''
        x = torch.flatten(x, start_dim=-2, end_dim=-1)      # [b,c,h,w]->[b,c,h*w]
        latent = self.linear1(x)

        x =self.linear2(latent)

        x = x.reshape(-1,1,28,28)
        x = self.decoder(x)

        # x:[6,3,224,224]
        # latent:[6,1,768]
        setsz=latent.size(0)
        latent = latent.view(setsz, -1)
        # latent:[6,768]
        loss=self.criterion(x,img)
        # 出来的张量为784*784

        # return loss,latent
        return x,latent,loss

    def infere(self, img):
        x0 = self.cnn_encoder(img)
        # encoder
        x, ftokens, attmaps = self.transformer.infere(x0)
        ftokens.insert(0, rearrange(x0, 'b c h w -> b (h w) c'))
        # decoder
        x = self.decoder(x)
        return x, ftokens, attmaps


class CNNEncoder2(nn.Module):
    def __init__(self, n_channels, out_channels, patch_height, patch_width):
        super(CNNEncoder2, self).__init__()
        self.scale = 1
        # CBR
        self.inc = SingleConv(n_channels, 64 // self.scale)
        self.down1 = DownSingleConv(64 // self.scale, 128 // self.scale)
        self.down2 = DownSingleConv(128 // self.scale, 256 // self.scale)
        self.down3 = DownSingleConv(256 // self.scale, out_channels)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x

class DownSingleConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            SingleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.CBR = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.CBR(x)


class CNNTransformer_record(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=784):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CNNAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches),
                CNNFeedForward(dim, mlp_dim, dropout=dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    def infere(self, x):
        ftokens, attmaps = [], []
        for attn, ff in self.layers:
            ax, amap = attn(x, mode="record")
            x = ax + x
            x = ff(x) + x
            ftokens.append(rearrange(x, 'b c h w -> b (h w) c'))
            attmaps.append(amap)
        return x, ftokens, attmaps


class CNNAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0., num_patches=784):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.num_patches = num_patches

        #self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=1, padding=0, bias=False)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=3, padding=1, bias=False)
        self.dis = relative_pos_dis(math.sqrt(num_patches), math.sqrt(num_patches), sita=0.9).to(args.device)
        self.headsita = nn.Parameter(torch.randn(heads), requires_grad=True)
        self.sig = nn.Sigmoid()

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim), # inner_dim
            nn.ReLU(inplace=True),
        ) if project_out else nn.Identity()

    def forward(self, x, mode="train", smooth=1e-4):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (g d) h w -> b g (h w) d', g=self.heads), qkv)
        attn = torch.matmul(q, k.transpose(-1, -2)) # b g n n
        qk_norm = torch.sqrt(torch.sum(q ** 2, dim=-1)+smooth)[:, :, :, None] * torch.sqrt(torch.sum(k ** 2, dim=-1)+smooth)[:, :, None, :] + smooth
        attn = attn/qk_norm
        # 记得注释掉下面一行
        # attentionheatmap_visual2(attn, self.sig(self.headsita), out_dir='./Visualization', value=1)
        factor = 1/(2*(self.sig(self.headsita)+0.01)**2) # h
        factor = 1/(2*(self.sig(self.headsita)*(0.4-0.003)+0.003)**2) # af3 + limited setting this, or using the above line code
        dis = factor[:, None, None]*self.dis[None, :, :] # g n n
        # print(dis)
        dis = torch.exp(-dis)
        dis = dis/torch.sum(dis, dim=-1)[:, :, None]

        # 记得注释掉
        # # 假设我们想要可视化dis张量的第一个层
        # layer_index = 0
        # attention_map = dis[layer_index, :, :]
        # attention_map = attention_map.cpu().numpy()
        # # 将注意力图下采样到更易于可视化的尺寸
        # # 注意力图可能非常大，因此我们使用matplotlib的imshow函数来显示它
        # plt.imshow(attention_map, cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.title("Attention Map Layer {}".format(layer_index))
        # plt.show()

        attn = attn * dis[None, :, :, :]
        # 记得注释掉
        # attentionheatmap_visual2(attn, self.sig(self.headsita), out_dir='./Visualization', value=0.003)

        #attentionheatmap_visual(attn, out_dir='./Visualization/attention_af3/')
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b g (h w) d -> b (g d) h w', h=x.shape[2])
        if mode=="train":
            return self.to_out(out)
        else:
            return self.to_out(out), attn

class CNNFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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
        return self.net(x)


def relative_pos_dis(height=32, weight=32, sita=0.9):
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww # 0 is 32 * 32 for h, 1 is 32 * 32 for w
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    dis = (relative_coords[:, :, 0].float()/height) ** 2 + (relative_coords[:, :, 1].float()/weight) ** 2
    #dis = torch.exp(-dis*(1/(2*sita**2)))
    return  dis

if __name__ == '__main__':
    tensor = torch.randn(6, 3, 224, 224).to("cuda")

    print(tensor.size())

    setr=CNNTransformer(3, 224)
    setr.to("cuda")
    x,latent=setr(tensor)
    print(x.shape,latent.size())
