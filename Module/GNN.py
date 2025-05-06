# -*-coding:utf-8-*-
import torch
from torch import nn

# Import commented out for modularity
# from Module.datautils import args
torch.autograd.set_detect_anomaly(True)


class GraphPropagationAttention(nn.Module):
    """
    Graph Propagation Attention module implementing a multi-head attention mechanism
    for graph structured data, supporting bidirectional information flow between nodes and edges.
    """

    def __init__(self, en_node_dim, en_edge_dim, node_dim, edge_dim, num_heads=4, qkv_bias=False, num_layers=3,
                 attn_drop=0., proj_drop=0.):
        """
        Initializes the Graph Propagation Attention module.

        Args:
            en_node_dim (int): Input dimension of node embeddings
            en_edge_dim (int): Input dimension of edge embeddings
            node_dim (int): Output dimension for node embeddings
            edge_dim (int): Output dimension for edge embeddings
            num_heads (int): Number of attention heads
            qkv_bias (bool): Whether to include bias in the query, key, value projections
            num_layers (int): Number of propagation layers
            attn_drop (float): Dropout rate for attention weights
            proj_drop (float): Dropout rate for output projections
        """
        super().__init__()
        # Linear projections for initial embedding transformations
        self.node_linear = nn.Sequential(nn.Linear(en_node_dim, node_dim), nn.LeakyReLU(0.1))
        self.edge_linear = nn.Sequential(nn.Linear(en_edge_dim, edge_dim), nn.LeakyReLU(0.1))

        self.num_heads = num_heads
        head_dim = en_node_dim // num_heads
        self.scale = head_dim ** -0.5  # Scaling factor for attention scores

        # Multi-head attention components
        self.qkv = nn.Linear(node_dim, node_dim * 3, bias=qkv_bias)  # Query, key, value projections
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(node_dim, node_dim)  # Output projection

        # Edge feature transformation layers
        self.reduce = nn.Conv2d(edge_dim, num_heads, kernel_size=1)  # Reduce edge features to attention biases
        self.expand = nn.Conv2d(num_heads, edge_dim, kernel_size=1)  # Expand attention weights back to edge features

        self.num_layers = num_layers

        # Handle dimension mismatch between edge and node features if needed
        if edge_dim != node_dim:
            self.fc = nn.Linear(edge_dim, node_dim)
        else:
            self.fc = nn.Identity()

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, node_embeds, edge_embeds, padding_mask):
        """
        Forward pass of the Graph Propagation Attention module.

        Args:
            node_embeds: Node embeddings tensor of shape [batch_size, num_nodes, node_embedding_dim]
            edge_embeds: Edge embeddings tensor of shape [batch_size, num_nodes, num_nodes, edge_embedding_dim]
            padding_mask: Padding mask tensor of shape [batch_size, num_nodes, 1]

        Returns:
            Updated node embeddings after graph propagation
        """
        # Transform input embeddings to the target dimensions
        node_embeds = self.node_linear(node_embeds)  # Shape: [B, N, node_dim]
        edge_embeds = self.edge_linear(edge_embeds)  # Shape: [B, N, N, edge_dim]

        # Multiple layers of graph propagation
        for _ in range(self.num_layers):
            B, N, C = node_embeds.shape  # B: batch size, N: number of nodes, C: node feature dimension

            # Reshape edge embeddings for attention computation
            edge_embeds = edge_embeds.reshape(B, N, N, -1).permute(0, 3, 1, 2)  # Shape: [B, edge_dim, N, N]
            padding_mask = padding_mask.reshape(B, N, 1).unsqueeze(1)  # Shape: [B, 1, N, 1]

            # Compute query, key, value projections for all heads
            qkv = self.qkv(node_embeds).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # qkv shape: [3, B, num_heads, N, head_dim]

            # Separate query, key, value tensors
            q, k, v = qkv.unbind(0)

            # Compute attention scores
            attn = (q @ k.transpose(-2, -1)) * self.scale  # Shape: [B, num_heads, N, N]

            # Add edge features as attention biases
            attn_bias = self.reduce(edge_embeds)  # Shape: [B, num_heads, N, N]
            attn = attn + attn_bias
            residual = attn  # Save for residual connection

            # Apply mask and softmax
            attn = attn.masked_fill(padding_mask, 0)  # Zero out padded positions
            attn = attn.softmax(dim=-1)  # Normalize attention weights
            attn = self.attn_drop(attn)

            # Apply attention to values
            node_embeds = (attn @ v).transpose(1, 2).reshape(B, N, C)  # Updated node features

            # Node-to-edge propagation: update edge features using attention weights
            edge_embeds = self.expand(attn + residual)  # Shape: [B, edge_dim, N, N]

            # Edge-to-node propagation: aggregate edge features to update nodes
            w = edge_embeds.masked_fill(padding_mask, 0)  # Apply padding mask
            w = w.softmax(dim=-1)  # Normalize weights
            w = (w * edge_embeds).sum(-1).transpose(-1, -2)  # Weighted aggregation

            # Add edge-to-node information to node embeddings
            node_embeds = node_embeds + self.fc(w)

            # Final projections and dropout
            node_embeds = self.proj(node_embeds)
            node_embeds = self.proj_drop(node_embeds)

        return node_embeds