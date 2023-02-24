import numpy as np
import os
import scipy.io
import math


def max_min_normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


class PreProcess(object):
    def __init__(self, atlas='', node_number=90, window_size=100, step=1, proportion=0.2,
                 data_dir='zhongda_data_dti'):
        # 需要传入的参数
        self.atlas = atlas  # '' '_ho' '_bn'
        self.node_number = node_number  # 90 113 246
        self.window_size = window_size  # 60:10:100
        self.step = step  # 1 2
        self.proportion = proportion
        # 由子类初始化的数据 1. 数据类型， 2. 相应类型标签
        self.subj_type_set = []
        self.subj_label_set = []
        # 常量参数数据
        self.main_dir = r'E:\多模态--graphCNN\data/' + data_dir + '/'
        # self.main_dir = '/home/wwh/graphcnn_zxt/data/' + data_dir + '/'
        # DTI_connectivity_count DTI_connectivity_log DTI_connectivity_ncount DTI_connectivity_voxel_norm
        # GQI_connectivity_count GQI_connectivity_log GQI_connectivity_ncount GQI_connectivity_voxel_norm
        # GQIConnectivity
        
        # self.DTI_connect_file_name = 'DTI_connectivity_count'
        self.DTI_connect_file_name = 'DTI_connectivity_new_liufuyuan'
        # self.DTI_connect_file_name = 'dti_FACT_45_02_1_1_Matrix_FA_AAL_Contract_90_2MM_90'
        self.DTI_feature_file_name = 'region_features_norm'
        self.fMRI_file_name = 'RegionSeries'
        self.total_window_size = 230
        self.adj_number = math.ceil((self.total_window_size - self.window_size + 1) / self.step)
        # 通过PreProcess获得的中间数据
        # 1. edge_weight_matrix ：DCN邻接矩阵；
        # 2. edge_weight_matrix_binary：通过proportion得到的二值化邻接矩阵
        # 3. node_label：每个节点的标签
        self.fMRI_vertices = []
        self.dynamic_fMRI_vertices = []
        self.DTI_vertices = []
        self.fMRI_adjacency = []
        self.dynamic_fMRI_adjacency = []
        self.DTI_adjacency = []
        self.fMRI_adjacency_binary = []
        self.dynamic_fMRI_adjacency_binary = []
        self.DTI_adjacency_binary = []
        self.gcn_label = []

    def compute_graph_cnn_input(self):
        self.compute_input_node_signal()
        return np.array(self.fMRI_vertices), np.array(self.fMRI_adjacency_binary), np.array(
            self.DTI_vertices), np.array(self.DTI_adjacency_binary), np.array(self.dynamic_fMRI_vertices), np.array(
            self.dynamic_fMRI_adjacency_binary), np.reshape(self.gcn_label, [-1])

    def compute_input_node_signal(self):
        self.compute_adjacency()
        self.compute_adjacency_binary()
        self.fMRI_adjacency_binary = np.expand_dims(self.fMRI_adjacency_binary, axis=2)
        self.DTI_adjacency_binary = np.expand_dims(self.DTI_adjacency_binary, axis=2)
        self.dynamic_fMRI_adjacency_binary = np.expand_dims(self.dynamic_fMRI_adjacency_binary, axis=3)

    # 对某一类别的样本计算edge_weight和label
    def compute_subj_adjacency(self, m_subj_dir, suj_label):
        subj_dir = os.listdir(m_subj_dir)
        # 对该类中每个样本计算特征
        for i in range(len(subj_dir)):
            subj_fMRI_file_name = m_subj_dir + '/' + subj_dir[i] + '/' + self.fMRI_file_name + self.atlas + '.mat'
            fMRI_vertices = scipy.io.loadmat(subj_fMRI_file_name)['RegionSeries']
            fMRI_adjacency = np.corrcoef(np.transpose(fMRI_vertices))
            subj_DTI_file_name = m_subj_dir + '/' + subj_dir[i] + '/' + self.DTI_feature_file_name + self.atlas + '.mat'
            DTI_vertices = scipy.io.loadmat(subj_DTI_file_name)['region_features']
            subj_DTI_file_name = m_subj_dir + '/' + subj_dir[i] + '/' + self.DTI_connect_file_name + self.atlas + '.mat'
            DTI_adjacency = scipy.io.loadmat(subj_DTI_file_name)['connectivity'][:90, :90]
            i_subj_adjacency = np.zeros((self.adj_number, self.node_number, self.node_number))
            i_subj_vertices = np.zeros((self.adj_number, self.node_number, self.window_size))
            for j in range(0, self.total_window_size - self.window_size, self.step):
                sub_vertices = fMRI_vertices[j:j + self.window_size, :]
                i_subj_vertices[math.ceil(j / self.step), :, :] = np.transpose(sub_vertices)
                i_subj_adjacency[math.ceil(j / self.step), :, :] = np.corrcoef(np.transpose(sub_vertices))
                # i_subj_adjacency[math.ceil(j / self.step), :, :] = DTI_adjacency
            self.fMRI_vertices.append(np.transpose(fMRI_vertices))
            self.fMRI_adjacency.append(fMRI_adjacency)
            self.DTI_vertices.append(DTI_vertices)
            self.DTI_adjacency.append(DTI_adjacency)
            self.dynamic_fMRI_vertices.append(i_subj_vertices)
            self.dynamic_fMRI_adjacency.append(i_subj_adjacency)
            self.gcn_label.append(suj_label)

    # 计算edge_weight_matrix和label
    def compute_adjacency(self):
        # 对所有类别数据计算edge_weight
        for i in range(len(self.subj_type_set)):
            # 获得数据类型，标签，数据读取路径
            i_subj_type = self.subj_type_set[i]
            i_subj_label = self.subj_label_set[i]
            main_subj_dir = self.main_dir + i_subj_type
            self.compute_subj_adjacency(main_subj_dir, i_subj_label)

    # 计算edge_weight_matrix_binary
    def compute_adjacency_binary(self):
        self.fMRI_adjacency = np.array(self.fMRI_adjacency)
        self.DTI_adjacency = np.array(self.DTI_adjacency)
        self.dynamic_fMRI_adjacency = np.array(self.dynamic_fMRI_adjacency)
        self.fMRI_adjacency_binary = np.zeros(self.fMRI_adjacency.shape)
        self.DTI_adjacency_binary = np.zeros(self.DTI_adjacency.shape)
        self.dynamic_fMRI_adjacency_binary = np.zeros(self.dynamic_fMRI_adjacency.shape)
        for i in range(self.fMRI_adjacency.shape[0]):
            self.fMRI_adjacency_binary[i, :, :] = self.compute_subj_adjacency_binary(self.fMRI_adjacency[i, :, :])
            self.DTI_adjacency_binary[i, :, :] = self.compute_subj_adjacency_binary(self.DTI_adjacency[i, :, :])
            for j in range(self.dynamic_fMRI_adjacency.shape[1]):
                self.dynamic_fMRI_adjacency_binary[i, j, :, :] = self.compute_subj_adjacency_binary(
                    self.dynamic_fMRI_adjacency[i, j, :, :])

    # 使用比例对单个邻接矩阵进行二值化计算，返回二值化矩阵
    def compute_subj_adjacency_binary(self, subj_edge_weight):
        edge_weight = subj_edge_weight - np.diag(np.diag(subj_edge_weight))
        edge_weight_list = self.mat_to_list(edge_weight)
        reserve_num = int(round(edge_weight_list.shape[1] * self.proportion))
        edge_weight_list_sorted = np.sort(edge_weight_list)
        threshold = edge_weight_list_sorted[0, -(reserve_num + 1)]
        edge_weight_binary = np.zeros((self.node_number, self.node_number))
        edge_weight_binary[edge_weight > threshold] = 1
        return edge_weight_binary

    # 取矩阵上三角，并转换为一维向量
    def mat_to_list(self, graph_matrix):
        graph_list = np.zeros((1, int(self.node_number * (self.node_number - 1) / 2)))
        index_start = 0
        for i in range(self.node_number - 1):
            index_end = index_start + self.node_number - i - 1
            graph_list[0, index_start: index_end] = graph_matrix[i, (i + 1): self.node_number]
            index_start = index_end
        return graph_list


# 继承于PreProcess，做HC_MDD分类
class HCMDDPreProcess(PreProcess):
    def __init__(self, atlas='', node_number=90, window_size=100, step=1, proportion=0.2,
                 data_dir='zhongda_data_fmri_dti'):
        super(HCMDDPreProcess, self).__init__(atlas, node_number, window_size, step, proportion, data_dir)
        # self.subj_type_set = ['HC', 'RMD', 'UMD']
        # self.subj_label_set = [0, 1, 1]
        
        self.subj_type_set = ['HC', 'MDD']
        self.subj_label_set = [0, 1]


# 继承于PreProcess，做分RD_NRD类
class RDNRDPreProcess(PreProcess):
    def __init__(self, atlas='', node_number=90, window_size=100, step=1, proportion=0.2,
                 data_dir='zhongda_data_fmri_dti'):
        super(RDNRDPreProcess, self).__init__(atlas, node_number, window_size, step, proportion, data_dir)
        self.subj_type_set = ['RMD', 'UMD']
        self.subj_label_set = [0, 1]


class XinXiangHCMDDPreProcess(PreProcess):
    def __init__(self, atlas='', node_number=90, window_size=100, step=1, proportion=0.2,
                 data_dir='xinxiang_data_fmri_dti'):
        super(XinXiangHCMDDPreProcess, self).__init__(atlas, node_number, window_size, step, proportion, data_dir)
        self.subj_type_set = ['HC', 'MDD']
        self.subj_label_set = [0, 1]


def compute_pooling_weight():
    node_sub_region_54 = np.array(
        [1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 7, 8, 7, 8, 7, 8, 7, 8, 9, 10, 11, 12, 13, 14, 13, 14, 15, 16, 17, 18, 19, 20,
         19, 20, 19, 20, 21, 22, 21, 22, 23, 24, 25, 26, 25, 26, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
         39, 40, 39, 40, 39, 40, 41, 42, 43, 44, 45, 46, 45, 46, 45, 46, 47, 48, 49, 50, 49, 50, 49, 50, 51, 52, 51, 52,
         53, 54])
    node_sub_region_14 = np.array(
        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 5, 6, 5,
         6, 5, 6, 5, 6, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 11,
         12, 11, 12, 11, 12, 11, 12, 13, 14, 13, 14, 13, 14, 13, 14, 13, 14, 13, 14])
    pooling_weight54 = np.zeros((90, 54))
    pooling_weight14 = np.zeros((54, 14))
    for i in range(90):
        pooling_weight54[i, node_sub_region_54[i] - 1] = 1
        pooling_weight14[node_sub_region_54[i] - 1, node_sub_region_14[i] - 1] = 1
    return pooling_weight54, pooling_weight14

# test = RDNRDPreProcess()
# dataset = test.compute_graph_cnn_input()
# print('end')
