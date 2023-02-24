import torch
import torch.nn as nn
import torch.nn.functional as F

import math


#2.构建self-Attention模块
class Attention(nn.Module):
    def __init__(self,args):
        super(Attention,self).__init__()

        self.num_attention_heads=args.num_heads #8
        self.attention_head_size = int(args.hidden_dim / self.num_attention_heads)  # 64/8=8
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 8*8=64

        self.query = nn.Linear(args.hidden_dim, self.all_head_size)#wm,64->64，Wq矩阵为（64,64）
        self.key = nn.Linear(args.hidden_dim, self.all_head_size)#wm,64->64,Wk矩阵为（64,64）
        self.value = nn.Linear(args.hidden_dim, self.all_head_size)#wm,64->64,Wv矩阵为（64->64）
        self.out = nn.Linear(args.hidden_dim, args.hidden_dim)  # wm,768->768
        self.dropout = args.dropout
        # self.attn_dropout = F.dropout(args.dropout)
        # self.proj_dropout = F.dropout(args.dropout)

        # self.softmax = F.log_softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size(-1) + (
        self.num_attention_heads, self.attention_head_size)  # wm,(bs,197)+(12,64)=(bs,197,12,64)
        x = x.view(*new_x_shape)
        return x.permute(1, 0, 2)  # wm,(bs,12,197,64)

    def forward(self, hidden_states):
        # hidden_states为：(bs,197,768)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.log_softmax(attention_scores, dim=-1)
        weights = attention_probs #wm,实际上就是权重
        attention_probs = F.dropout(attention_probs, p=self.dropout, training=self.training)

        context_layer = torch.matmul(attention_probs, value_layer)#将概率与内容向量相乘
        context_layer = context_layer.permute(1, 0, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-1] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = F.dropout(attention_output, p=self.dropout, training=self.training)
        return attention_output, weights

