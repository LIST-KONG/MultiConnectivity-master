import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch_geometric.nn import GCNConv, global_sort_pool


class DGCNN(torch.nn.Module):
    def __init__(self, args):
        super(DGCNN, self).__init__()
        self.num_classes = args.num_classes
        self.num_features = args.sc_features
        self.hidden = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropuot = args.dropout
        self.k = 30
        self.conv1 = GCNConv(self.num_features, self.hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden, self.hidden))
        self.conv1d = Conv1d(self.hidden, 32, 5)

        self.lin1 = Linear(32 * (self.k - 5 + 1), self.hidden)
        self.lin2 = Linear(self.hidden, self.hidden // 2)
        self.lin3 = Linear(self.hidden // 2, self.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_sort_pool(x, batch, self.k)
        x = x.view(len(x), self.k, -1).permute(0, 2, 1)
        x = F.relu(self.conv1d(x))
        x = x.view(len(x), -1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropuot, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropuot, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
