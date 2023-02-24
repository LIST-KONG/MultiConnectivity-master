import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_transformer.Attention import Attention
from graph_transformer.Mlp import Mlp

# 4.构建编码器的可重复利用的Block()模块：每一个block包含了self-attention模块和MLP模块
class Block(nn.Module):
    def __init__(self, args):
        super(Block, self).__init__()
        self.hidden_size = args.hidden_dim
        # self.attention_norm = nn.functional.normalize(self.hidden_size, dim=1)  # wm，层归一化
        # self.ffn_norm = nn.functional.normalize(self.hidden_size, dim=1)

        self.ffn = Mlp(args)
        self.attn = Attention(args)

    def forward(self, x):
        h = x
        x = nn.functional.normalize(x,dim=1)
        x, weights = self.attn(x)
        x = x +  h  # 残差结构

        h = x
        x = nn.functional.normalize(x,dim=1)
        x = self.ffn(x)
        x = x + h  # 残差结构
        return x, weights
