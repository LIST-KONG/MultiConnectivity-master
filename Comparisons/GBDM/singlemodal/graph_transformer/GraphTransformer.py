import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_transformer.Transformer import Transformer


#7构建VisionTransformer，用于图像分类
class GraphTransformer(nn.Module):
    def __init__(self, args):
        super(GraphTransformer, self).__init__()
        self.args = args
        self.num_classes = args.num_classes
        self.Transformer = Transformer(self.args)

        # self.transformer = Transformer(args)
        self.head = nn.Linear(args.hidden_dim, self.num_classes)#wm,768-->10

    def forward(self, data):
        sc_x = data.x
        sc_x = Transformer(sc_x)

        return sc_x

