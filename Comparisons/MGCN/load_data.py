import os
import numpy as np
import torch
from scipy import io
from construct import get_Pearson_fc,get_BOLD_feature,get_Degree_fc,get_sc,get_sc_feature,get_dataloader, get_fc_edgeWeight, get_sc_dataloader, get_fc_dataloader
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from scipy import sparse as sp
import dgl
import hashlib


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def decide_dataset(root):  # 返回标签
    if 'ABIDE' in root:
        class_dict = {
            "HC": 0,
            "ASD": 1,
        }
    elif 'multi' in root or "xinxiang" in root or "zhongda" in root:
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


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    
    return lap_pos_enc


def wl_positional_encoding(g):
    """
        WL-based absolute positional embedding 
        adapted from 
        
        "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
        Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
        https://github.com/jwzhanggy/Graph-Bert
    """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1

    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1
        
    wl_pos_enc = torch.LongTensor(list(node_color_dict.values()))
    return wl_pos_enc

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
                sc_data = get_sc_dataloader(subj_sc_adj, subj_sc_feature)
                sc_data.y = label
                self.sc.append(sc_data)
                ############
                fc_timeseries = read_dataset_fc_regionSeries(root, label_files, files)  # 这里得到的已经是节点数*长度的时间序列
                # fc_edgeWeight = read_dataset_fc_edgeWeight(root, label_files, files)
                fc_region_features = read_dataset_fc_region_features(root, label_files, files)
                subj_fc_adj = get_Pearson_fc(fc_timeseries,
                                                   args)  # 这里传参是时间序列，返回的是邻接矩阵，以方法命名，因为可能有不同的构造方法，例如度矩阵等construct_Degree_fc;threshold等参数通过args传递
                subj_fc_feature = get_fc_edgeWeight(fc_region_features,
                                                         args)  # 同样传参时间序列，返回特征，注意这里特征和上面邻接矩阵都返回的是矩阵，构造Data放在后面，以防有的会直接使用这两个矩阵，可以将Data构造部分注释掉
                # 构造fc Data
                fc_data = get_dataloader(subj_fc_adj, subj_fc_feature)
                fc_data.y = label
                self.fc.append(fc_data)
                ############
                # 构造sc fc Data
                data = get_sc_dataloader(subj_sc_adj, subj_sc_feature)
                data.y = label
                data.fc_x = get_dataloader(subj_fc_adj, subj_fc_feature).x
                data.fc_edge_index = get_dataloader(subj_fc_adj, subj_fc_feature).edge_index
                self.scfc.append(data)
                self.y.append(label)

        self.choose_data = self.scfc
       
# 多中心数据
class multi_center:

    # 加载单中心fc数据
    def load_one_site(self, center_path):
        label_set = {'HC': 0, 'MDD': 1}
        label_list = os.listdir(center_path)  # HC, MDD
        fc = []

        threshold = 0.2
        for label_files in label_list:
            list_path = os.path.join(center_path, label_files)
            lists = os.listdir(list_path)  # .mat文件
            for i in range(len(lists)):
                mat_path = os.path.join(list_path, lists[i])
                fc_data = center_fc_construct(mat_path)
                fc_data.y = torch.tensor(label_set[label_files])
                fc.append(fc_data)


        return fc

    # 加载多中心数据
    def load_multi_site_MDD(self, path):
        main_path = path
        center_dir = os.listdir(main_path)
        datasets = []
        for center in center_dir:
            center_data_path = os.path.join(main_path, center)
            one_site_ds = self.load_one_site(center_data_path)
            datasets.append(one_site_ds)
        return datasets


#graph transfomer 模型数据加载
class FSDataset_GT:
    def __init__(self, args):
        self.sc = []
        self.fc = []
        self.scfc = []
        self.class_dict = decide_dataset(args.path)
        self.y = []

        root = args.path
        label_list = os.listdir(root)
        label_list.sort()
        maxx = 0

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
                maxx = max(maxx, subj_sc_feature.max())
                # 构造sc Data
                sc_data = get_sc_dataloader(subj_sc_adj, subj_sc_feature)
                sc_data.y = label
                self.sc.append(sc_data)
                ############
                fc_timeseries = read_dataset_fc_regionSeries(root, label_files, files)
                fc_edgeWeight = read_dataset_fc_edgeWeight(root, label_files, files)
                subj_fc_adj = get_Pearson_fc(fc_timeseries,
                                                   args)
                subj_fc_feature = get_fc_edgeWeight(fc_edgeWeight,
                                                         args)
                # 构造fc Data
                fc_data = get_fc_dataloader(subj_fc_adj, subj_fc_feature)
                fc_data.y = label
                fc_data.fc_g = get_fc_dataloader(subj_fc_adj, subj_fc_feature).g
                if args.lap_pos_enc:
                    # Graph positional encoding v/ Laplacian eigenvectors
                    fc_data.lap_pos_enc = laplacian_positional_encoding(fc_data.fc_g, args.pos_enc_dim)
                if args.wl_pos_enc:
                    # WL positional encoding from Graph-Bert, Zhang et al 2020.
                    fc_data.wl_pos_enc = wl_positional_encoding(fc_data.g)
                self.fc.append(fc_data)
                ############
                # 构造sc fc Data
                data = get_sc_dataloader(subj_sc_adj, subj_sc_feature)
                data.adj = subj_sc_adj
                data.number_of_nodes = data.adj.shape[0]
                data.y = label
                data.fc_x = get_fc_dataloader(subj_fc_adj, subj_fc_feature).x
                data.fc_edge_index = get_fc_dataloader(subj_fc_adj, subj_fc_feature).edge_index
                data.fc_g = get_fc_dataloader(subj_fc_adj, subj_fc_feature).g
                if args.lap_pos_enc:
                    # Graph positional encoding v/ Laplacian eigenvectors
                    data.lap_pos_enc = laplacian_positional_encoding(data.g, args.pos_enc_dim)
                    data.fc_lap_pos_enc = laplacian_positional_encoding(data.fc_g, args.pos_enc_dim)
                if args.wl_pos_enc:
                    # WL positional encoding from Graph-Bert, Zhang et al 2020.
                    data.wl_pos_enc = wl_positional_encoding(data.g)
                self.scfc.append(data)
                self.y.append(label)

        self.choose_data = self.scfc
        print(maxx)

