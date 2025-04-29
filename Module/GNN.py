# -*-coding:utf-8-*-
import torch
from torch import nn
#
# from Module.datautils import args
torch.autograd.set_detect_anomaly(True)

class GraphPropagationAttention(nn.Module):
    def __init__(self, en_node_dim, en_edge_dim,node_dim,edge_dim, num_heads=4, qkv_bias=False,num_layers=3, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.node_linear=nn.Sequential(nn.Linear(en_node_dim, node_dim), nn.LeakyReLU(0.1))

        self.edge_linear=nn.Sequential(nn.Linear(en_edge_dim, edge_dim), nn.LeakyReLU(0.1))
        self.num_heads = num_heads
        head_dim = en_node_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(node_dim, node_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(node_dim, node_dim)

        self.reduce = nn.Conv2d(edge_dim, num_heads, kernel_size=1)
        self.expand = nn.Conv2d(num_heads, edge_dim, kernel_size=1)
        self.num_layers = num_layers
        if edge_dim != node_dim:
            self.fc = nn.Linear(edge_dim, node_dim)
        else:
            self.fc = nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)

   # s_node_embed:torch.Size([6, 251, 31])
   # s_edge_embed:torch.Size([6, 251, 251, 6])
    #s_img_embed:torch.Size([6, 3, 224, 224])
    #padding_mask_s:torch.Size([6, 251, 1])
    def forward(self, node_embeds, edge_embeds, padding_mask):

        node_embeds= self.node_linear(node_embeds)   # torch.Size([6, 251, 256])
        # edge_embeds=self.edge_linear(edge_embeds).permute(0,1,4,2,3)
        edge_embeds=self.edge_linear(edge_embeds)   # torch.Size([6, 251, 251, 64])

        for _ in range(self.num_layers):
            B, N, C = node_embeds.shape  # node_embeds: batch,Node Count,Embedding Dimension    这里的B是setsize
            # B =batch*setsize
            # node_embeds=node_embeds.view(B,N,C)
            edge_embeds=edge_embeds.reshape(B,N,N,-1).permute(0,3,1,2) # [b,c,n,n]
            padding_mask=padding_mask.reshape(B,N,1).unsqueeze(1)     # [b,1,n,1]

            qkv = self.qkv(node_embeds).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # qkv.shape=torch.size([24,3,8,50,64])
 
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, n_head, 1+N, 1+N]
            attn_bias = self.reduce(edge_embeds)  # [B, C, 1+N, 1+N] -> [B, n_head, 1+N, 1+N]
            attn = attn + attn_bias  # [B, n_head, 1+N, 1+N]
            residual = attn

            # attn = attn.masked_fill(padding_mask, float("-inf"))
            attn = attn.masked_fill(padding_mask, 0)
            attn = attn.softmax(dim=-1)  # [B, C, N, N]
            attn = self.attn_drop(attn)
            node_embeds = (attn @ v).transpose(1, 2).reshape(B, N, C)

            # node-to-edge propagation
            edge_embeds = self.expand(attn + residual)  # [B, n_head, 1+N, 1+N] -> [B, C, 1+N, 1+N]

            # edge-to-node propagation
            w = edge_embeds.masked_fill(padding_mask, 0)
            w = w.softmax(dim=-1)
            w = (w * edge_embeds).sum(-1).transpose(-1, -2)
            node_embeds = node_embeds + self.fc(w)
            node_embeds = self.proj(node_embeds)
            node_embeds= self.proj_drop(node_embeds)


        return node_embeds