import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GraphConv, ASAPooling, JumpingKnowledge
from torch_geometric.nn import global_add_pool


class ASAP(torch.nn.Module):
    def __init__(self, args, ratio=0.8):
        super(ASAP, self).__init__()
        self.num_classes = args.num_classes
        self.features = args.sc_features
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.hidden = args.hidden_dim
        self.conv1 = GraphConv(self.features, self.hidden, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(self.hidden, self.hidden, aggr='mean')
            for i in range(self.num_layers - 1)
        ])
        self.pools.extend([
            ASAPooling(self.hidden, ratio, dropout=self.dropout)
            for i in range(self.num_layers // 2)
        ])

        self.jump = JumpingKnowledge(mode='cat')

        self.lin1 = Linear(self.num_layers * self.hidden * 2, self.hidden)
        self.lin2 = Linear(self.hidden, self.hidden // 2)
        self.lin3 = Linear(self.hidden // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = None
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_add_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            xs += [global_add_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, edge_weight, batch, _ = pool(
                    x=x, edge_index=edge_index, edge_weight=edge_weight,
                    batch=batch)
        x = self.jump(xs)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__