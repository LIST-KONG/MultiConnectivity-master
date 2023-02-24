from torch_geometric.nn import GCNConv
from torch.nn import Parameter
from torch.nn import Linear
import torch.nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GCNConv, GATConv, TAGConv, ChebConv
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
from torch_geometric.nn import GraphConv
from torch_geometric.nn import BatchNorm
from MMTopKpy import TopKPooling, MMTopKPool

class MMGNN(torch.nn.Module):
    def __init__(self,args):
        super(MMGNN, self).__init__()
        #torch.manual_seed(12345)
        
        self.args = args
        ratio = 0.8

        self.conv1 = GraphConv(args.sc_features, args.hidden_dim)
        self.conv2 = GraphConv(args.fc_features, args.hidden_dim)

        #choose desired multimodal pooling layer here
        self.pool_double1 = MMTopKPool(args.hidden_dim, ratio = ratio)
        #self.pool_double1 = MMPool(args.hidden_dim, ratio = ratio, num_ROIs = 273)

        self.conv3 = GraphConv(args.hidden_dim, args.hidden_dim)
        self.conv4 = GraphConv(args.hidden_dim, args.hidden_dim)

        self.norm1 = BatchNorm(args.hidden_dim)
        self.norm2 = BatchNorm(args.hidden_dim)

        self.norm3 = BatchNorm(args.hidden_dim)
        self.norm4 = BatchNorm(args.hidden_dim)

        self.norm5 = BatchNorm(int(4*args.hidden_dim))

        self.lin_single = Linear(int(2*args.hidden_dim), int(args.num_classes))
        self.lin_double = Linear(int(4*args.hidden_dim), int(args.num_classes))

    def forward(self, data):
        #print(x, batch)
        x1, edge_index1, batch = data.x, data.edge_index, data.batch
        x2, edge_index2 = data.fc_x, data.fc_edge_index
        edge_attr1 = None
        edge_attr2 = None
        x1 = self.conv1(x1, edge_index1)#, edge_weight = edge_attr1)
        x2 = self.conv2(x2, edge_index2)#, edge_weight = edge_attr2)

        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        
        x1 = x1.relu()
        x2 = x2.relu()

        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.dropout(x2, p=0.2, training=self.training)

        x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2, batch, perm, score, _, _ = self.pool_double1(x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2, batch = batch)

        x1 = self.conv3(x1, edge_index1)#, edge_weight = edge_attr1)
        x2 = self.conv4(x2, edge_index2)#, edge_weight = edge_attr2)
        #print(x1[0])

        x1 = self.norm3(x1)
        x2 = self.norm4(x2)
        
        x1 = x1.relu()
        x2 = x2.relu()

        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.dropout(x2, p=0.2, training=self.training)


        x1 = torch.cat([global_max_pool(x1, batch), global_mean_pool(x1, batch)], dim=1)
        x2 = torch.cat([global_max_pool(x2, batch), global_mean_pool(x2, batch)], dim=1)

        x = torch.cat((torch.atleast_2d(x1),torch.atleast_2d(x2)),dim = 1)

        x = self.norm5(x)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin_double(x)
        # x = torch.softmax(x, 1).squeeze(1)
        x = F.log_softmax(x, dim=-1)

        
        # return x, score

        return x
