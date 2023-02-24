import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import torch.nn as nn
import numpy as np


class GIN(torch.nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()
        self.num_classes = args.num_classes
        self.sc_features = args.sc_features
        self.fc_features = args.fc_features
        self.num_layers = args.num_layers
        self.hidden = args.hidden_dim
        self.dropout = args.dropout
        self.temperature = args.temperature
        self.negative_w = args.negative_weight
        self.sc_conv1 = GINConv(
            Sequential(
                Linear(self.sc_features, self.hidden),
                ReLU(),
                Linear(self.hidden, self.hidden),
                ReLU(),
                BN(self.hidden),
            ), train_eps=True)
        self.sc_convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.sc_convs.append(
                GINConv(
                    Sequential(
                        Linear(self.hidden, self.hidden),
                        ReLU(),
                        Linear(self.hidden, self.hidden),
                        ReLU(),
                        BN(self.hidden),
                    ), train_eps=True))

        self.fc_conv1 = GINConv(
            Sequential(
                Linear(self.fc_features, self.hidden),
                ReLU(),
                Linear(self.hidden, self.hidden),
                ReLU(),
                BN(self.hidden),
            ), train_eps=True)
        self.fc_convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.fc_convs.append(
                GINConv(
                    Sequential(
                        Linear(self.hidden, self.hidden),
                        ReLU(),
                        Linear(self.hidden, self.hidden),
                        ReLU(),
                        BN(self.hidden),
                    ), train_eps=True))

        self.fc1 = nn.Linear(self.hidden * 2, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden // 2)
        self.fc3 = nn.Linear(self.hidden // 2, self.num_classes)

    def fc_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)

        return x

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

        # return x

    def __repr__(self):
        return self.__class__.__name__