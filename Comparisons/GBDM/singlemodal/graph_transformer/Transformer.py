import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_transformer.Encoder import Encoder
from graph_transformer.Embeddings import Embeddings

#6构建transformers完整结构，首先图片被embedding模块编码成序列数据，然后送入Encoder中进行编码
class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        self.embeddings = Embeddings(args)
        self.encoder = Encoder(args)
        self.hidden_dim = args.hidden_dim

    def forward(self):
        # sc_x = data.x;
        # embeddings = self.embeddings(sc_x)
        embedding_output = self.embeddings
        encoded = self.encoder(embedding_output)
        return encoded


