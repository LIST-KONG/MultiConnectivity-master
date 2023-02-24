import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (GraphConv, TopKPooling, global_add_pool,
                                JumpingKnowledge)


class GraphUnet(torch.nn.Module):
    def __init__(self, args, ratio=0.8):
        super(GraphUnet, self).__init__()
        self.num_classes = args.num_classes
        self.num_features = args.sc_features
        self.hidden = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropuot = args.dropout

        self.conv1 = GraphConv(self.num_features, self.hidden, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(self.hidden, self.hidden, aggr='mean')
            for i in range(self.num_layers - 1)
        ])
        self.pools.extend(
            [TopKPooling(self.hidden, ratio) for i in range((self.num_layers) // 2)])
        self.jump = JumpingKnowledge(mode='cat')

        self.lin1 = Linear(self.num_layers * self.hidden, self.hidden)
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
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_add_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            xs += [global_add_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, _, batch, _, _ = pool(x, edge_index,
                                                     batch=batch)
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropuot, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropuot, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
