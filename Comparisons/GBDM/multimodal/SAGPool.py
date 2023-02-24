import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn
import numpy as np
from torch_geometric.nn import (GraphConv, SAGPooling, global_add_pool,
                                JumpingKnowledge)


class SAGPool(torch.nn.Module):
    def __init__(self, args, ratio=0.8):
        super(SAGPool, self).__init__()
        self.num_classes = args.num_classes
        self.sc_features = args.sc_features
        self.fc_features = args.fc_features
        self.hidden = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropuot = args.dropout
        self.temperature = args.temperature
        self.negative_w = args.negative_weight
        self.sc_conv1 = GraphConv(self.sc_features, self.hidden, aggr='mean')
        self.sc_convs = torch.nn.ModuleList()
        self.sc_pools = torch.nn.ModuleList()
        self.sc_convs.extend([
            GraphConv(self.hidden, self.hidden, aggr='mean')
            for i in range(self.num_layers - 1)
        ])
        self.sc_pools.extend(
            [SAGPooling(self.hidden, ratio) for i in range(self.num_layers // 2)])
        self.sc_jump = JumpingKnowledge(mode='cat')

        self.fc_conv1 = GraphConv(self.fc_features, self.hidden, aggr='mean')
        self.fc_convs = torch.nn.ModuleList()
        self.fc_pools = torch.nn.ModuleList()
        self.fc_convs.extend([
            GraphConv(self.hidden, self.hidden, aggr='mean')
            for i in range(self.num_layers - 1)
        ])
        self.fc_pools.extend(
            [SAGPooling(self.hidden, ratio) for i in range(self.num_layers // 2)])
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

    def compute_loss(self, logits, mask):
        return - torch.log( (F.softmax(logits, dim=1) * mask).sum(1) )

    def _get_positive_mask(self, batch_size):
        diag = np.eye(batch_size)
        mask = torch.from_numpy((diag))
        mask = (1 - mask)
        return mask.cuda(non_blocking=True)

    def forward(self, data):
        sc_x, sc_edge_index, batch = data.x, data.edge_index, data.batch
        sc_x = F.relu(self.sc_conv1(sc_x, sc_edge_index))
        sc_xs = [global_add_pool(sc_x, batch)]
        for i, conv in enumerate(self.sc_convs):
            sc_x = F.relu(conv(sc_x, sc_edge_index))
            sc_xs += [global_add_pool(sc_x, batch)]
            if i % 2 == 0 and i < len(self.sc_convs) - 1:
                pool = self.sc_pools[i // 2]
                sc_x, sc_edge_index, _, batch, _, _ = pool(sc_x, sc_edge_index,
                                                     batch=batch)
        sc_x = self.sc_jump(sc_xs)

        fc_x, fc_edge_index = data.fc_x, data.fc_edge_index
        fc_x = F.relu(self.fc_conv1(fc_x, fc_edge_index))
        fc_xs = [global_add_pool(fc_x, batch)]
        for i, conv in enumerate(self.fc_convs):
            fc_x = F.relu(conv(fc_x,fc_edge_index))
            fc_xs += [global_add_pool(fc_x, batch)]
            if i % 2 == 0 and i < len(self.fc_convs) - 1:
                pool = self.fc_pools[i // 2]
                fc_x, fc_edge_index, _, batch, _, _ = pool(fc_x, fc_edge_index,
                                                     batch=batch)
        fc_x = self.fc_jump(fc_xs)
        x = torch.cat([sc_x, fc_x], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropuot, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropuot, training=self.training)
        x = self.lin3(x)
        # return F.log_softmax(x, dim=-1)

        batch_size = sc_x.shape[0]

        # Normalize features
        sc_x = nn.functional.normalize(sc_x, dim=1)
        fc_x = nn.functional.normalize(fc_x, dim=1)

        # Inter-modality alignment
        logits_per_sc = sc_x @ fc_x.t()
        logits_per_fc = fc_x @ sc_x.t()

        # Intra-modality alignment
        logits_clstr_sc = sc_x @ sc_x.t()
        logits_clstr_fc = fc_x @ fc_x.t()

        logits_per_sc /= self.temperature
        logits_per_fc /= self.temperature
        logits_clstr_sc /= self.temperature
        logits_clstr_fc /= self.temperature

        positive_mask = self._get_positive_mask(sc_x.shape[0])
        negatives_sc = logits_clstr_sc * positive_mask
        negatives_fc = logits_clstr_fc * positive_mask

        sc_logits = torch.cat([logits_per_sc, self.negative_w * negatives_sc], dim=1)
        fc_logits = torch.cat([logits_per_fc, self.negative_w * negatives_fc], dim=1)

        diag = np.eye(batch_size)
        mask_sc = torch.from_numpy((diag)).cuda()
        mask_fc = torch.from_numpy((diag)).cuda()

        mask_neg_s = torch.zeros_like(negatives_sc)
        mask_neg_f = torch.zeros_like(negatives_fc)
        mask_s = torch.cat([mask_sc, mask_neg_s], dim=1)
        mask_f = torch.cat([mask_fc, mask_neg_f], dim=1)

        loss_i = self.compute_loss(sc_logits, mask_s)
        loss_t = self.compute_loss(fc_logits, mask_f)

        return F.log_softmax(x, dim=-1) + (((loss_i.mean() + loss_t.mean())) / 2)


    def __repr__(self):
        return self.__class__.__name__
