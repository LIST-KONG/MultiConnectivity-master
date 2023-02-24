import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (GraphConv, TopKPooling, global_add_pool,
                                JumpingKnowledge)
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

import dgl

"""
    Graph Transformer with edge features
    
"""
from multimodal.graph_transformer.layers.graph_transformer_edge_layer import GraphTransformerLayer
from multimodal.graph_transformer.layers.graph_transformer_multimodal_sc_layer import GraphTransformer_Multimodal_sc_Layer
from multimodal.graph_transformer.layers.graph_transformer_multimodal_fc_layer import GraphTransformer_Multimodal_fc_Layer
from multimodal.graph_transformer.layers.mlp_readout_layer import MLPReadout

class GraphTransformerNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        num_atom_type = args.num_atom_type
        hidden_dim = args.hidden_dim
        num_heads = args.num_heads
        out_dim = args.hidden_dim
        in_feat_dropout = args.in_feat_dropout
        dropout = args.dropout
        n_layers = args.num_layers
        ratio = 0.8
        self.sc_features = args.sc_features
        self.fc_features = args.fc_features
        self.hidden = args.hidden_dim
        self.num_layers = args.num_layers
        self.readout = args.readout
        self.lap_pos_enc = args.lap_pos_enc
        self.layer_norm = args.layer_norm
        self.batch_norm = args.batch_norm
        self.residual = args.residual
        self.device = args.device
        self.num_classes = args.num_classes
        self.dropout = args.dropout

        self.embedding_lap_pos_enc = nn.Linear(args.pos_enc_dim, hidden_dim)

        self.embedding_h = nn.Linear(hidden_dim, hidden_dim)

        self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.sc_conv1 = GCNConv(self.sc_features, self.hidden, aggr='mean')
        self.sc_convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.sc_convs.append(GCNConv(self.hidden, self.hidden, aggr='mean'))
        self.sc_lin = Linear(self.hidden , self.hidden)

        self.fc_conv1 = GCNConv(self.fc_features, self.hidden, aggr='mean')
        self.fc_convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.fc_convs.append(GCNConv(self.hidden, self.hidden, aggr='mean'))
        self.fc_lin = nn.Linear(self.hidden , self.hidden)
        
        self.sc_layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ])
        self.sc_layers.append(GraphTransformer_Multimodal_sc_Layer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.fc_layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ])
        self.fc_layers.append(GraphTransformer_Multimodal_fc_Layer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        # self.MLP_layer = MLPReadout(n_layers*out_dim*2, 2)
        self.lin1 = Linear(self.num_layers * self.hidden * 2, self.hidden)
        self.lin2 = Linear(self.hidden, self.hidden // 2)
        self.lin3 = Linear(self.hidden // 2, self.num_classes)
        
    def forward(self, data):# h_lap_pos_enc=None):

        sc_x, sc_edge_index, batch = data.x, data.edge_index, data.batch
        fc_x, fc_edge_index = data.fc_x, data.fc_edge_index
        sc_g = dgl.batch(data.g).to(self.device)
        fc_g = dgl.batch(data.fc_g).to(self.device)

        sc_x = F.relu(self.sc_conv1(sc_x, sc_edge_index))
        for conv in self.fc_convs:
            sc_x = F.relu(conv(sc_x, sc_edge_index))
        sc_x = self.sc_lin(sc_x)

        fc_x = F.relu(self.fc_conv1(fc_x, fc_edge_index))
        for conv in self.fc_convs:
            fc_x = F.relu(conv(fc_x, fc_edge_index))
        fc_x = self.fc_lin(fc_x)

        if self.lap_pos_enc:
            h_lap_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(h_lap_pos_enc.size(1)).to(self.device).cuda()
            sign_flip[sign_flip>=0.5] = 1.0
            sign_flip[sign_flip<0.5] = -1.0
            h_lap_pos_enc = h_lap_pos_enc * sign_flip.unsqueeze(0).to(self.device).cuda()

            fc_h_lap_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(fc_h_lap_pos_enc.size(1)).to(self.device).cuda()
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            fc_h_lap_pos_enc = fc_h_lap_pos_enc * sign_flip.unsqueeze(0).to(self.device).cuda()
        else:
            h_lap_pos_enc = None

        # input embedding
        sc_h = self.embedding_h(sc_x.float())
        sc_h = self.in_feat_dropout(sc_h)
        sc_h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
        sc_h = sc_h + sc_h_lap_pos_enc
        sc_e = torch.ones(sc_edge_index.size(1),1).to(self.device)
        sc_e = self.embedding_e(sc_e)

        fc_h = self.embedding_h(fc_x.float())
        fc_h = self.in_feat_dropout(fc_h)
        fc_h_lap_pos_enc = self.embedding_lap_pos_enc(fc_h_lap_pos_enc.float())
        fc_h = fc_h + fc_h_lap_pos_enc
        fc_e = torch.ones(fc_edge_index.size(1),1).to(self.device)
        fc_e = self.embedding_e(fc_e)

        
        # convnets
        # sc_hs = [global_add_pool(sc_h, batch)]
        for conv in self.sc_layers:
            sc_h, sc_e = conv(sc_g, sc_h, fc_h, sc_e)
        # sc_hs += [global_add_pool(sc_h, batch)]
        sc_g.ndata['h'] = sc_h

        # fc_hs = [global_add_pool(fc_h, batch)]
        for conv in self.fc_layers:
            fc_h, fc_e = conv(fc_g, fc_h, sc_h, fc_e)
        # fc_hs += [global_add_pool(fc_h, batch)]
        fc_g.ndata['h'] = fc_h

        if self.readout == "sum":
            sc_hg = dgl.sum_nodes(sc_g, 'h')
        elif self.readout == "max":
            sc_hg = dgl.max_nodes(sc_g, 'h')
        elif self.readout == "mean":
            sc_hg = dgl.mean_nodes(sc_g, 'h')
        else:
            sc_hg = dgl.mean_nodes(sc_g, 'h')  # default readout is mean nodes

        if self.readout == "sum":
            fc_hg = dgl.sum_nodes(fc_g, 'h')
        elif self.readout == "max":
            fc_hg = dgl.max_nodes(fc_g, 'h')
        elif self.readout == "mean":
            fc_hg = dgl.mean_nodes(fc_g, 'h')
        else:
            fc_hg = dgl.mean_nodes(fc_g, 'h')  # default readout is mean nodes

        # sc_h = self.sc_jump(sc_hs)
        # fc_h = self.fc_jump(fc_hs)

        # h = torch.cat([sc_h, fc_h], dim=1)
        hg = torch.cat([sc_hg, fc_hg], dim=1)
            
        # return F.log_softmax(self.MLP_layer(h), dim=-1)
        x = F.relu(self.lin1(hg))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)
        
    # def loss(self, scores, targets):
    #     # loss = nn.MSELoss()(scores,targets)
    #     loss = nn.L1Loss()(scores, targets)
    #     return loss
