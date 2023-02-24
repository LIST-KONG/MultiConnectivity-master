import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_transformer.Block import Block


#5.构建Encoder模块，该模块实际上就是堆叠N个Block模块
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.num_layers = args.num_layers
        self.layer = nn.ModuleList()
        # self.encoder_norm = nn.functional.normalize(hidden_states, dim=1)
        for i in range(self.num_layers - 1):
            self.layer.append( Block(args) )

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            attn_weights.append(weights)
        encoded = nn.functional.normalize(hidden_states, dim=1)
        return encoded
