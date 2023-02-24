import os
import numpy as np
import torch
import math
from scipy import io
from construct import get_Pearson_fc,get_BOLD_feature,get_Degree_fc,get_sc,get_sc_feature,get_dataloader, get_fc_edgeWeight, get_sc_dataloader, get_fc_dataloader
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from scipy import sparse as sp
import hashlib


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def decide_dataset(root):  # 返回标签
    if 'ABIDE' in root:
        class_dict = {
            "HC": 0,
            "ASD": 1,
        }
    elif 'Multi' in root or "xinxiang" in root or "zhongda" or "multi_SCFC" in root:
        class_dict = {
            "HC": 0,
            "MDD": 1,
        }
    elif "HCP" in root:
        class_dict = {
            "female": 0,
            "male": 1,
        }
    return class_dict


def read_dataset_fc_regionSeries(root, label_files, files):  # 返回时间序列
    FC_dir = "RegionSeries.mat"
    if 'ABIDE' in root:
        subj_fc_dir = os.path.join(root, label_files, files)
        subj_mat_fc = np.loadtxt(subj_fc_dir)[:176, :90]
    elif 'Multi' in root:
        subj_fc_dir = os.path.join(root, label_files, files)
        subj_mat_fc = io.loadmat(subj_fc_dir)['ROISignals_AAL'][:170, :]
    else:  # zhonda、xinxiang、ASD的读法都一样
        subj_fc_dir = os.path.join(root, label_files, files, FC_dir)
        subj_mat_fc = io.loadmat(subj_fc_dir)['RegionSeries']
    return subj_mat_fc

def read_dataset_fc_edgeWeight(root, label_files, files):  # 返回时间序列
    FC_dir = "EdgeWeight.mat"
    subj_fc_dir = os.path.join(root, label_files, files, FC_dir)
    subj_mat_fc = io.loadmat(subj_fc_dir)['EdgeWeight']
    return subj_mat_fc

def read_dataset_fc_region_features(root, label_files, files):  # 返回时间序列
    FC_dir = "region_features_norm.mat"
    subj_fc_dir = os.path.join(root, label_files, files, FC_dir)
    subj_mat_fc = io.loadmat(subj_fc_dir)['region_features']
    return subj_mat_fc

def read_dataset_sc_connectivity(root, label_files, files):  # 返回sc connectivity
    SC_connectivity_dir = "dti_FACT_45_02_1_1_Matrix_FA_AAL_Contract_90_2MM_90.txt"
    subj_sc_connectivity_dir = os.path.join(root, label_files, files, SC_connectivity_dir)
    subj_sc_connectivity = np.loadtxt(subj_sc_connectivity_dir)
    # subj_sc_connectivity = io.loadmat(subj_sc_connectivity_dir)['connectivity']
    # subj_sc_connectivity = subj_sc_connectivity[0:90, 0:90]
    return subj_sc_connectivity

def read_dataset_sc_features(root, label_files, files):  # 返回sc特征
    SC_features_dir = "region_features_norm.mat"
    subj_sc_feature_dir = os.path.join(root, label_files, files, SC_features_dir)
    subj_sc_feature = io.loadmat(subj_sc_feature_dir)['region_features']
    return subj_sc_feature

def read_dataset_fc_region_features(root, label_files, files):  # 返回时间序列
    FC_dir = "region_features_norm.mat"
    subj_fc_dir = os.path.join(root, label_files, files, FC_dir)
    subj_mat_fc = io.loadmat(subj_fc_dir)['region_features']
    return subj_mat_fc

def d_constraint(X, D, a):
    n, d = X.shape
    sum_dist = 0

    for i in range(n):
        for j in range(i+1, n):
            if D[i, j] == 1:
                d_ij = X[i] - X[j]
                dist_ij = distance1(a, d_ij)
                sum_dist += dist_ij
    fD = gf(sum_dist)
    return fD


def gf(sum_dist):
    fD = np.log(sum_dist)
    return fD


def distance1(a, d_ij):
    fudge = 0.000001
    dist_ij = np.sqrt(np.dot(d_ij**2, a))  # distance between X[i] and X[j]
    return dist_ij


class FSDataset:
    def __init__(self, args):
        self.sc = []
        self.fc = []
        self.scfc = []
        self.class_dict = decide_dataset(args.path)
        self.y = []

        root = args.path
        label_list = os.listdir(root)
        label_list.sort()

        for label_files in label_list:
            list = os.listdir(os.path.join(root, label_files))
            list.sort()
            label = torch.LongTensor([self.class_dict[label_files]])
            for files in list:
                ############
                sc_connectivity = read_dataset_sc_connectivity(root, label_files, files)
                sc_feature = read_dataset_sc_connectivity(root, label_files, files)
                subj_sc_adj = get_sc(sc_connectivity, args)  # 构造邻接矩阵
                subj_sc_feature = get_sc_feature(sc_feature, args)  # 构造特征
                # 构造sc Data
                # sc_data = get_sc_dataloader(subj_sc_adj, subj_sc_feature)
                # sc_data.y = label
                # self.sc.append(sc_data)
                ############
                fc_timeseries = read_dataset_fc_regionSeries(root, label_files, files)  # 这里得到的已经是节点数*长度的时间序列
                subj_fc_adj = get_Pearson_fc(fc_timeseries,
                                                   args)
                # fc_edgeWeight = read_dataset_fc_edgeWeight(root, label_files, files)
                d = np.ones((90, 90)) - np.tril(np.ones((90, 90)))
                a = np.ones(90)
                re = d_constraint( subj_fc_adj, d, a)
                re = math.exp(-(re/2*args.bandwidth*args.bandwidth))
                # subj_fc_adj = get_Pearson_fc(fc_timeseries,
                #                                    args)  # 这里传参是时间序列，返回的是邻接矩阵，以方法命名，因为可能有不同的构造方法，例如度矩阵等construct_Degree_fc;threshold等参数通过args传递
                fc_region_features = read_dataset_fc_region_features(root, label_files, files)
                subj_fc_feature = re * get_fc_edgeWeight(subj_fc_adj ,
                                                        args)  # 同样传参时间序列，返回特征，注意这里特征和上面邻接矩阵都返回的是矩阵，构造Data放在后面，以防有的会直接使用这两个矩阵，可以将Data构造部分注释掉
                # subj_fc_feature = re * subj_fc_adj
                # 构造fc Data
                # fc_data = get_dataloader(subj_fc_adj, subj_fc_feature)
                # fc_data.y = label
                # self.fc.append(fc_data)
                ############
                # 构造sc fc Data

                # self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
                # self.fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
                # self.fuse_weight_1.data.fill_(1)
                # self.fuse_weight_2.data.fill_(1)
                feature = np.identity(90) +  subj_fc_feature + subj_sc_feature
                adj = np.identity(90)

                data = get_dataloader(adj, feature)
                data.y = label
                # data.fc_x = get_dataloader(subj_fc_adj, subj_fc_feature).x
                # data.fc_edge_index = get_dataloader(subj_fc_adj, subj_fc_feature).edge_index
                self.scfc.append(data)
                self.y.append(label)

        self.choose_data = self.scfc
