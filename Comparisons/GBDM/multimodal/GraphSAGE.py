import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class GraphSAGE(nn.Module):
    def __init__(self, args):
        super(GraphSAGE, self).__init__()
        self.args = args
        self.sc_features = args.sc_features
        self.fc_features = args.fc_features
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.num_classes = args.num_classes
        self.dropout = args.dropout

        self.sc_conv1 = SAGEConv(self.sc_features, self.hidden_dim)
        self.sc_convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.sc_convs.append(SAGEConv(self.hidden_dim, self.hidden_dim))

        self.fc_conv1 = SAGEConv(self.fc_features, self.hidden_dim)
        self.fc_convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.fc_convs.append(SAGEConv(self.hidden_dim, self.hidden_dim))

        self.fc1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_classes)

    def fc_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)

        return x

    def forward(self, data):
        sc_x, sc_edge_index, batch = data.x, data.edge_index, data.batch
        sc_x = F.relu(self.sc_conv1(sc_x, sc_edge_index))
        for conv in self.sc_convs:
            sc_x = F.relu(conv(sc_x, sc_edge_index))
        sc_x = global_add_pool(sc_x, batch)

        fc_x, fc_edge_index = data.fc_x, data.fc_edge_index
        fc_x = F.relu(self.fc_conv1(fc_x, fc_edge_index))
        for conv in self.fc_convs:
            fc_x = F.relu(conv(fc_x, fc_edge_index))
        fc_x = global_add_pool(fc_x, batch)

        x = torch.cat([sc_x, fc_x], dim=1)
        x = self.fc_forward(x)

        return x

    def __repr__(self):
        return self.__class__.__name__
