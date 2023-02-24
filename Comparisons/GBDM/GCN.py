import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        self.features = args.sc_features
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.num_classes = args.num_classes
        self.dropout = args.dropout

        self.conv1 = GCNConv(self.features, self.hidden_dim, aggr='mean')
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim, aggr='mean'))

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_classes)

        self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_1.data.fill_(1)
        self.fuse_weight_2.data.fill_(1)

    def fc_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)

        return x

    def forward(self, data):
        # sc_x, edge_index, batch = data.x, data.edge_index, data.batch
        # fc_x, fc_edge_index = data.fc_x, data.fc_edge_index

        # x = np.identity(90) +  self.fuse_weight_1*sc_x + self.fuse_weight_2*fc_x

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_add_pool(x, batch)
        x = self.fc_forward(x)

        return x

    def __repr__(self):
        return self.__class__.__name__
