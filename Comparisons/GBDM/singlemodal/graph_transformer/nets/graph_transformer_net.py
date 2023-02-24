import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features
    
"""
from singlemodal.graph_transformer.layers.graph_transformer_edge_layer import GraphTransformerLayer
from singlemodal.graph_transformer.layers.mlp_readout_layer import MLPReadout

class GraphTransformerNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_atom_type = args.num_atom_type
        num_edge_type = args.num_edge_type
        hidden_dim = args.hidden_dim
        num_heads = args.num_heads
        out_dim = args.hidden_dim
        in_feat_dropout = args.in_feat_dropout
        dropout = args.dropout
        n_layers = args.num_layers
        self.readout = args.readout
        self.lap_pos_enc = args.lap_pos_enc
        self.layer_norm = args.layer_norm
        self.batch_norm = args.batch_norm
        self.residual = args.residual
        self.device = args.device

        self.embedding_lap_pos_enc = nn.Linear(args.pos_enc_dim, hidden_dim)

        self.embedding_h = nn.Linear(num_atom_type, hidden_dim)

        self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.fc_layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ])
        self.fc_layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim*2, 2)   # 1 out dim since regression problem
        
    def forward(self, data):# h_lap_pos_enc=None):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        g = dgl.batch(data.g).to(self.device)

        if self.lap_pos_enc:
            h_lap_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(h_lap_pos_enc.size(1)).to(self.device).cuda()
            sign_flip[sign_flip>=0.5] = 1.0
            sign_flip[sign_flip<0.5] = -1.0
            h_lap_pos_enc = h_lap_pos_enc * sign_flip.unsqueeze(0).to(self.device).cuda()
        else:
            h_lap_pos_enc = None

        # input embedding
        h = self.embedding_h(x.float())
        h = self.in_feat_dropout(h)
        h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
        h = h + h_lap_pos_enc
        e = torch.ones(edge_index.size(1),1).to(self.device)
        e = self.embedding_e(e)
        
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return F.log_softmax(self.MLP_layer(hg), dim=-1)
        
    # def loss(self, scores, targets):
    #     # loss = nn.MSELoss()(scores,targets)
    #     loss = nn.L1Loss()(scores, targets)
    #     return loss
