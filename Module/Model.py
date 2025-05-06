# -*-coding:utf-8-*-
# Unified molecular prediction model based on meta-learning
import torch
import torch.nn as nn
from Module.AttentiveLayers import CNNTransformer
from Module.GNN import GraphPropagationAttention
from Module.datautils import args


class UMPredict(nn.Module):
    """
    Unified Molecular Prediction model combining CNN and Graph Neural Network approaches.
    Integrates image and molecular graph data for comprehensive molecular property prediction.
    """

    def __init__(self, en_node_dim, en_edge_dim, p_dropout):
        """
        Initialize the Unified Molecular Prediction model.

        Args:
            en_node_dim (int): Input dimension of node features
            en_edge_dim (int): Input dimension of edge features
            p_dropout (float): Dropout probability for regularization
        """
        super(UMPredict, self).__init__()
        self.en_node_dim = en_node_dim
        self.en_edge_dim = en_edge_dim
        self.p_dropout = p_dropout

        # CNN-Transformer for image processing
        self.cnn = CNNTransformer(n_channels=args.img_channels, imgsize=args.imgsize)

        # Graph Propagation Attention for molecular graph processing
        self.gpa = GraphPropagationAttention(
            en_node_dim=en_node_dim,
            en_edge_dim=en_edge_dim,
            node_dim=args.node_dim,
            edge_dim=args.edge_dim,
            qkv_bias=True
        )

        # Loss function
        self.criterion = nn.BCELoss()

        # Final prediction layers
        # Note: input features combine node embeddings and image embeddings
        self.last_linear1 = nn.Linear(in_features=args.node_dim + args.img_dim, out_features=64, bias=True)
        self.last_linear2 = nn.Linear(64, 1, bias=True)

        # Activation function
        self.sigmoid = nn.Sigmoid()

        # Freeze CNN model parameters to prevent updating during training
        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, atom_features, edge_features, padding_mask, img_tensor, y_target):
        """
        Forward pass through the unified molecular prediction model.

        Args:
            atom_features (torch.Tensor): Node/atom features of shape [setsz, 251, 31]
            edge_features (torch.Tensor): Edge features of shape [setsz, 251, 251, 6]
            padding_mask (torch.Tensor): Padding mask for variable-sized molecular graphs
            img_tensor (torch.Tensor): Image tensor of shape [setsz, 3, 224, 224]
            y_target (torch.Tensor): Target labels

        Returns:
            tuple: (loss, predictions, targets) - Model loss, predicted values, and target values
        """
        # Process molecular graph with Graph Propagation Attention
        node_embeddings = self.gpa(atom_features, edge_features, padding_mask).to(args.device)

        # Get batch size (set size in this context)
        setsize = node_embeddings.shape[0]

        # For simplicity, only use the first node embedding as graph-level representation
        # Extract first node from each graph as representative embedding
        node_embeddings = node_embeddings[:, 0, :]
        node_embeddings = node_embeddings.reshape(setsize, -1)

        # Process image data through CNN-Transformer
        # Expected to return (reconstructed_image, latent_representation, reconstruction_loss)
        _, latent, _ = self.cnn(img_tensor)
        latent = latent.reshape(setsize, -1)

        # Concatenate graph and image embeddings
        concatenated = torch.cat((node_embeddings.reshape(setsize, -1), latent.reshape(setsize, -1)), dim=1)

        # Final prediction layers
        output = self.last_linear1(concatenated)
        output = self.sigmoid(output)
        predict = self.sigmoid(self.last_linear2(output)).reshape(-1)

        # Calculate loss using binary cross-entropy
        loss = self.criterion(predict, y_target)

        # Return loss, predictions and targets with consistent data type
        return loss.to(dtype=torch.float32), predict.to(dtype=torch.float32), y_target.to(dtype=torch.float32)