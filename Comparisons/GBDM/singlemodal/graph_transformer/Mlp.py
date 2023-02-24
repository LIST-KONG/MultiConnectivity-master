import torch
import torch.nn as nn
import torch.nn.functional as F

#3.构建前向传播神经网络
#两个全连接神经网络，中间加了激活函数
class Mlp(nn.Module):
    def __init__(self, args):
        super(Mlp, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.num_classes = args.num_classes
        self.dropout = args.dropout

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)#wm,786->3072
        self.fc2 = nn.Linear( self.hidden_dim // 2, args.num_classes)#wm,3072->786
        self.act_fn = F.relu#wm,激活函数
        # self.dropout = F.dropout(args.dropout)

    def forward(self, x):
        x = self.fc1(x)#wm,786->3072
        x = self.act_fn(x)#激活函数
        x = F.dropout(x, p=self.dropout, training=self.training)#wm,丢弃
        x = self.fc2(x)#wm3072->786
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
