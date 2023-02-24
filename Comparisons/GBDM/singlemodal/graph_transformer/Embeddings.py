import torch
import torch.nn as nn
import torch.nn.functional as F

class Embeddings(nn.Module):
    '''
    对图像进行编码，把图片当做一个句子，把图片分割成块，每一块表示一个单词
    '''
    def __init__(self, args):
        super(Embeddings ,self).__init__()
        self.args = args
        self.sc_features = args.sc_features
        self.fc_features = args.fc_features
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout


        self.sc_embedding =nn.Linear(self.sc_features, self.hidden_dim)
        # self.fc_embedding = nn.Linear(self.fc_features, self.hidden_dim)

        # 位置编码信息
        self.position_embedding =nn.Linear(self.sc_features, self.hidden_dim)

    def forward(self, data):
        sc_x = data.x
        sc_x = self.sc_embedding(self.sc_x);
        embeddings=sc_x+self.position_embedding# 将图片块信息  和对其位置信息进行相加
        embeddings=F.dropout(embeddings, p=self.dropout, training=self.training)
        return  embeddings

