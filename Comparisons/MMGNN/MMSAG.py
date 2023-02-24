from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter
import torch
from torch.nn import Linear
from torch.nn import Conv1d
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GCNConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
from torch_geometric.nn import GraphConv
from torch_geometric.nn import TopKPooling
#from torch_geometric.nn import SAGPooling
from torch_geometric.nn import BatchNorm

#based on the code found at: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/pool/sag_pool.html#SAGPooling
class MMPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh, enforce_same_score = False, num_ROIs = 273):
        super(MMPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio

        self.hardcoded_score1 = torch.nn.Parameter(torch.zeros(num_ROIs))
        self.hardcoded_score2 = torch.nn.Parameter(torch.zeros(num_ROIs))
        #self.hardcoded_score3 = torch.nn.Parameter(torch.normal(torch.zeros(273), torch.zeros(273) + 0.1), requires_grad = True)

        self.score_layer1 = Conv(in_channels,1)
        self.score_layer2 = Conv(in_channels,1)
        self.non_linearity = non_linearity
        self.group_score = Conv1d(2,1,1)
        self.enforce_same_score = enforce_same_score

    def forward(self, x1,x2, edge_index1, edge_index2, edge_attr1 = None, edge_attr2 = None, batch=None):
        if batch is None:
            batch = edge_index1.new_zeros(x1.size(0))
        
        if self.enforce_same_score == False:
        # THIS IS THE ORIGINAL
            score1 = self.score_layer1(x1,edge_index1).squeeze()
            score2 = self.score_layer2(x2,edge_index2).squeeze()

            score1 = self.non_linearity(score1)
            score2 = self.non_linearity(score2)

            score = torch.stack((score1,score2)).unsqueeze(0)
            score = self.group_score(score).squeeze()

        else:
            score1 = self.hardcoded_score1.repeat(len(torch.unique(batch)))
            score2 = self.hardcoded_score2.repeat(len(torch.unique(batch)))
            score = torch.stack((score1,score2)).unsqueeze(0)
            score = self.group_score(score).squeeze()
    

        perm = topk(score, self.ratio, batch)

        x1 = x1[perm] * self.nonlinearity(score[perm]).view(-1, 1)
        x2 = x2[perm] * self.nonlinearity(score[perm]).view(-1, 1)

        batch = batch[perm]

        edge_index1, edge_attr1 = filter_adj(edge_index1, edge_attr1, perm, num_nodes=score.size(0))
        edge_index2, edge_attr2 = filter_adj(edge_index2, edge_attr2, perm, num_nodes=score.size(0))

        return x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2, batch, perm, score, score1, score2



class SAGPool_ROI(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh, enforce_same_score = False, num_ROIs = 273):
        super(SAGPool_ROI,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
        self.hardcoded_score = torch.nn.Parameter(torch.zeros(num_ROIs))
        self.enforce_same_score = enforce_same_score

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x

        if self.enforce_same_score == False:
            score = self.score_layer(x,edge_index).squeeze()
        else:
            score = score = self.hardcoded_score3.repeat(len(torch.unique(batch)))

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score
