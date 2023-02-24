import os
import numpy as np
import torch
from scipy import io
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
import dgl

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

##########################################
def get_Pearson_fc(fc_timeseries,args):
    threshold=args.threshold
    fc_adj = np.corrcoef(np.transpose(fc_timeseries))
    fc_adj_list = fc_adj.reshape((-1))
    thindex = int(threshold * fc_adj_list.shape[0])
    thremax = fc_adj_list[fc_adj_list.argsort()[-1 * thindex]]
    fc_adj[fc_adj < thremax] = 0
    fc_adj[fc_adj >= thremax] = 1
    return fc_adj

def get_BOLD_feature(fc_timeseries,args):
    fc_list = fc_timeseries.reshape((-1))
    fc_new = (fc_timeseries - min(fc_list)) / (
            max(fc_list) - min(fc_list))
    fc_new = np.transpose(fc_new)
    return fc_new

def get_fc_edgeWeight(fc_edgeWeight,args):
    fc_list = fc_edgeWeight.reshape((-1))
    fc_new = (fc_edgeWeight - min(fc_list)) / (
            max(fc_list) - min(fc_list))
    return fc_new

def get_fc_region_features(fc_region_features,args):
    fc_list = fc_region_features.reshape((-1))
    fc_new = (fc_region_features - min(fc_list)) / (
            max(fc_list) - min(fc_list))
    return fc_new

def get_Degree_fc(fc_timeseries,args):
    fc_adj = get_Pearson_fc(fc_timeseries,args)
    degree_matrix =np.zeros((90,90))
    colsum=fc_adj.sum(axis=0)
    for j in range(fc_adj.shape[0]):
        degree_matrix[j][j] = colsum[j]
    return degree_matrix

def get_sc(sc_adj,args):
    threshold=args.threshold
    #构造sc邻接矩阵
    sc_adj_list = sc_adj.reshape((-1))
    thindex = int(threshold * sc_adj_list.shape[0])
    thremax = sc_adj_list[sc_adj_list.argsort()[-1 * thindex]]
    sc_adj[sc_adj < thremax] = 0
    sc_adj[sc_adj >= thremax] = 1
    return sc_adj

def get_sc_feature(sc_region_features,args):
    #构造sc feature
    sc_feature = sc_region_features
    return sc_feature

def get_dataloader(adj, feature):
    edge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
    x = torch.from_numpy(feature.astype(np.int16)).float()
    # g = dgl.graph((edge_index[0], edge_index[1]))
    data = Data(x=x, edge_index=edge_index)
    return data

def get_sc_dataloader(adj, feature):
    edge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
    x = torch.from_numpy(feature.astype(np.int16)).float()
    data = Data(x=x, edge_index=edge_index)
    return data

def get_fc_dataloader(adj, feature):
    edge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
    x = torch.from_numpy(feature.astype(np.int16)).float()
    data = Data(x=x, edge_index=edge_index)
    return data

##########################################