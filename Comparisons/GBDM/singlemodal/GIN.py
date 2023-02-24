import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class GIN(torch.nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()
        self.num_classes = args.num_classes
        self.features = args.sc_features
        self.hidden = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.conv1 = GINConv(
            Sequential(
                Linear(self.features, self.hidden),
                ReLU(),
                Linear(self.hidden, self.hidden),
                ReLU(),
                BN(self.hidden),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(self.hidden, self.hidden),
                        ReLU(),
                        Linear(self.hidden, self.hidden),
                        ReLU(),
                        BN(self.hidden),
                    ), train_eps=True))

        self.fc1 = nn.Linear(self.hidden, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden // 2)
        self.fc3 = nn.Linear(self.hidden // 2, self.num_classes)

    def fc_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)

        return x

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_add_pool(x, batch)

        x = self.fc_forward(x)

        return x

    def __repr__(self):
        return self.__class__.__name__