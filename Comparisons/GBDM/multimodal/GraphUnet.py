import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (GraphConv, TopKPooling, global_add_pool,
                                JumpingKnowledge)


class GraphUnet(torch.nn.Module):
    def __init__(self, args, ratio=0.8):
        super(GraphUnet, self).__init__()
        self.num_classes = args.num_classes
        self.sc_features = args.sc_features
        self.fc_features = args.fc_features
        self.hidden = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropuot = args.dropout

        self.sc_conv1 = GraphConv(self.sc_features, self.hidden, aggr='mean')
        self.sc_convs = torch.nn.ModuleList()
        self.sc_pools = torch.nn.ModuleList()
        self.sc_convs.extend([
            GraphConv(self.hidden, self.hidden, aggr='mean')
            for i in range(self.num_layers - 1)
        ])
        self.sc_pools.extend(
            [TopKPooling(self.hidden, ratio) for i in range(self.num_layers // 2)])
        self.sc_jump = JumpingKnowledge(mode='cat')

        self.fc_conv1 = GraphConv(self.fc_features, self.hidden, aggr='mean')
        self.fc_convs = torch.nn.ModuleList()
        self.fc_pools = torch.nn.ModuleList()
        self.fc_convs.extend([
            GraphConv(self.hidden, self.hidden, aggr='mean')
            for i in range(self.num_layers - 1)
        ])
        self.fc_pools.extend(
            [TopKPooling(self.hidden, ratio) for i in range((self.num_layers) // 2)])
        self.fc_jump = JumpingKnowledge(mode='cat')

        self.lin1 = Linear(self.num_layers * self.hidden * 2, self.hidden)
        self.lin2 = Linear(self.hidden, self.hidden // 2)
        self.lin3 = Linear(self.hidden // 2, self.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data):
        sc_x, sc_edge_index, batch = data.x, data.edge_index, data.batch
        sc_x = F.relu(self.sc_conv1(sc_x, sc_edge_index))
        sc_xs = [global_add_pool(sc_x, batch)]
        for i, conv in enumerate(self.sc_convs):
            sc_x = F.relu(conv(sc_x, sc_edge_index))
            sc_xs += [global_add_pool(sc_x, batch)]
            if i % 2 == 0 and i < len(self.sc_convs) - 1:
                sc_pool = self.sc_pools[i // 2]
                sc_x, sc_edge_index, _, batch, _, _ = sc_pool(sc_x, sc_edge_index,
                                                              batch=batch)
        sc_x = self.sc_jump(sc_xs)

        fc_x, fc_edge_index = data.fc_x, data.fc_edge_index
        fc_x = F.relu(self.fc_conv1(fc_x, fc_edge_index))
        fc_xs = [global_add_pool(fc_x, batch)]
        for i, conv in enumerate(self.fc_convs):
            fc_x = F.relu(conv(fc_x, fc_edge_index))
            fc_xs += [global_add_pool(fc_x, batch)]
            if i % 2 == 0 and i < len(self.fc_convs) - 1:
                fc_pool = self.fc_pools[i // 2]
                fc_x, fc_edge_index, _, batch, _, _ = fc_pool(fc_x, fc_edge_index,
                                                              batch=batch)
        fc_x = self.fc_jump(fc_xs)
        x = torch.cat([sc_x, fc_x], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropuot, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropuot, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
