from typing import Union, Optional, Callable
import torch
from torch.nn import Parameter
from torch_scatter import scatter_add, scatter_max
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch_geometric.utils import softmax
from torch.nn import Conv1d
from torch.nn.init import uniform_, uniform

#based on the code found at: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/pool/topk_pool.html#TopKPooling

class MMTopKPool(torch.nn.Module):
    def __init__(self, in_channels: int, ratio: Union[int, float] = 0.5,
                 min_score: Optional[float] = None, multiplier: float = 1.,
                 nonlinearity: Callable = torch.tanh):
        super(MMTopKPool, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.weight1 = Parameter(torch.Tensor(1, in_channels))
        self.weight2 = Parameter(torch.Tensor(1, in_channels))

        self.group_score = Conv1d(2,1,1)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform_(self.weight1)
        uniform_(self.weight2)

    def forward(self, x1, x2, edge_index1, edge_index2, edge_attr1 = None, edge_attr2 = None, batch=None, attn=None):
        """"""

        if batch is None:
            batch = edge_index.new_zeros(x1.size(0))

        attn1 = x1.unsqueeze(-1) if x1.dim() == 1 else x1
        score1 = (attn1 * self.weight1).sum(dim=-1)

        attn2 = x2.unsqueeze(-1) if x2.dim() == 1 else x2
        score2 = (attn2 * self.weight2).sum(dim=-1)
        

        if self.min_score is None:
            score1 = self.nonlinearity(score1 / self.weight1.norm(p=2, dim=-1))
            score2 = self.nonlinearity(score2 / self.weight2.norm(p=2, dim=-1))
        else:
            score1 = softmax(score1, batch)
            score2 = softmax(score2, batch)

        score = torch.stack((score1,score2)).unsqueeze(0)

        score = self.group_score(score).squeeze()

        perm = topk(score, self.ratio, batch, self.min_score)

        x1 = x1[perm] * self.nonlinearity(score[perm]).view(-1, 1)
        x2 = x2[perm] * self.nonlinearity(score[perm]).view(-1, 1)

        x1 = self.multiplier * x1 if self.multiplier != 1 else x1
        x2 = self.multiplier * x2 if self.multiplier != 1 else x2

        batch = batch[perm]

        edge_index1, edge_attr1 = filter_adj(edge_index1, edge_attr1, perm, num_nodes=score.size(0))
        edge_index2, edge_attr2 = filter_adj(edge_index2, edge_attr2, perm, num_nodes=score.size(0))

        return x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2, batch, perm, score, score1, score2

class TopKPooling(torch.nn.Module):
    def __init__(self, in_channels: int, ratio: Union[int, float] = 0.5,
                 min_score: Optional[float] = None, multiplier: float = 1.,
                 nonlinearity: Callable = torch.tanh):
        super(TopKPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.weight = Parameter(torch.Tensor(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform_(self.weight)


    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        """"""

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = (attn * self.weight).sum(dim=-1)

        if self.min_score is None:
            score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score
