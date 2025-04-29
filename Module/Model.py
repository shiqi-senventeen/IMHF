# -*-coding:utf-8-*-
# Unified molecular prediction model based on meta-learning
import torch
import torch.nn as nn

from Module.AttentiveLayers import CNNTransformer
from Module.GNN import GraphPropagationAttention
from Module.datautils import args


class UMPredict(nn.Module):
    def __init__(self, en_node_dim, en_edge_dim,p_dropout):
        super(UMPredict, self).__init__()
        self.en_node_dim = en_node_dim
        self.en_edge_dim = en_edge_dim
        self.p_dropout = p_dropout

        self.cnn = CNNTransformer(n_channels=args.img_channels, imgsize=args.imgsize)
        self.gpa = GraphPropagationAttention(en_node_dim=en_node_dim, en_edge_dim=en_edge_dim, node_dim=args.node_dim,
                                             edge_dim=args.edge_dim, qkv_bias=True)
        # self.fuse_linear = nn.Linear(in_features=1024, out_features=384)
        # self.fc = nn.Linear(384, out_features=2)
        self.criterion = nn.BCELoss()
        # 这里的in_feature需要调整
        self.last_linear1 = nn.Linear(in_features=args.node_dim+args.img_dim, out_features=64,bias=True)
        self.last_linear2 = nn.Linear(64, 1,bias=True)
        # self.alpha = nn.Parameter(torch.randn(1))
        # self.beta = nn.Parameter(torch.randn(1))
        # self.gamma = nn.Parameter(torch.randn(1))
        self.sigmoid = nn.Sigmoid()
        # self.relu=nn.ReLU()

        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, atom_features, edge_features, padding_mask, img_tensor, y_target):
        # node:[batch,setsz,201,768]
        # latent:[batch,setsz,1,768]
        # atom_features:[setsz,251,31]
        # edge_features:[setsz,251,251,6]
        node_embeddings = self.gpa(atom_features, edge_features, padding_mask).to(args.device)
        # indices = torch.zeros(201, dtype=torch.long)
       
        setsize = node_embeddings.shape[0]
        node_embeddings = node_embeddings[:, 0, :]
        node_embeddings=node_embeddings.reshape(setsize, -1)
        _, latent ,_= self.cnn(img_tensor)
        latent=latent.reshape(setsize, -1)
        # latent.reshape(setsize,-1)

        concatenated = torch.cat((node_embeddings.reshape(setsize, -1), latent.reshape(setsize, -1)), dim=1)

        # output = self.sigmoid(self.last_linear1(concatenated))
        # output = self.relu(self.last_linear1(concatenated))
        # concatenated=torch.cat((self.alpha*x,self.beta*node_embeddings,self.gamma*latent),dim=1)
        output=self.last_linear1(concatenated)
        output=self.sigmoid(output)

        predict = self.sigmoid(self.last_linear2(output)).reshape(-1)
        loss = self.criterion(predict, y_target)


        # concatenated = torch.cat((latent, selected_node), dim=2)
        # linear_output=self.linear(concatenated)
        #
        # pooled = torch.mean(linear_output, dim=1)
        # logits = self.fc(pooled)
        # predict= self.sigmoid(logits)
        # loss=self.criterion(predict,y_target)+loss1

        return loss.to(dtype=torch.float32), predict.to(dtype=torch.float32), y_target.to(dtype=torch.float32)
        # return loss1, predict.to(dtype=torch.float32),y_target.to(dtype=torch.float32)
