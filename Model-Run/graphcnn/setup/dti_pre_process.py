import numpy as np
import os
import scipy.io
import math


def max_min_normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


class PreProcess(object):
    def __init__(self, atlas='', node_number=90, window_size=100, step=1, proportion=0.2, data_dir='zhongda_data_dti'):
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
        self.main_dir = 'E:/Code/fMRI_preprocess/' + data_dir + '/'
        self.dti_file_name = 'DTIConnectivity_norm'  # DTIConnectivity DTIConnectivity_norm GQIConnectivity
        self.total_window_size = 230
        self.adj_number = math.ceil((self.total_window_size - self.window_size + 1) / self.step)
        # 通过PreProcess获得的中间数据
        # 1. edge_weight_matrix ：DCN邻接矩阵；
        # 2. edge_weight_matrix_binary：通过proportion得到的二值化邻接矩阵
        # 3. node_label：每个节点的标签
        self.dti_edge_weight_matrix = []
        self.dti_edge_weight_matrix_binary = []
        self.gcn_node_label_signal_matrix = []
        # gcn输入数据
        # 1. gcn_label：标签
        # 2. gcn_node_label_matrix： 节点特征
        # 3. gcn_edge_feature：邻接矩阵
        self.gcn_label = []
        self.gcn_node_feature = []
        self.gcn_edge_feature = []

    def compute_graph_cnn_input(self, if_binary_edge=True, if_proportion=True):
        self.compute_input_node_signal(if_binary_edge, if_proportion)
        return np.array(self.gcn_node_feature), np.array(self.gcn_edge_feature), np.reshape(self.gcn_label, [-1])

    def compute_input_node_signal(self, if_binary_edge=True, if_proportion=True):
        self.compute_dti_edge_weight()
        if if_proportion:
            self.compute_dti_edge_weight_binary_proportion()
        else:
            self.compute_dti_edge_weight_binary()
        if if_binary_edge:
            self.gcn_edge_feature = self.dti_edge_weight_matrix_binary
        else:
            self.gcn_edge_feature = np.multiply(self.dti_edge_weight_matrix_binary, self.dti_edge_weight_matrix)
        self.gcn_edge_feature = np.expand_dims(self.gcn_edge_feature, axis=3)
        self.gcn_node_feature = self.gcn_node_label_signal_matrix

    # 计算edge_weight_matrix和label
    def compute_dti_edge_weight(self):
        # 对所有类别数据计算edge_weight
        for i in range(len(self.subj_type_set)):
            # 获得数据类型，标签，数据读取路径
            i_subj_type = self.subj_type_set[i]
            i_subj_label = self.subj_label_set[i]
            main_subj_dir = self.main_dir + i_subj_type
            self.compute_subj_dti_edge_weight(main_subj_dir, i_subj_label)
        self.dti_edge_weight_matrix = np.array(self.dti_edge_weight_matrix)
        self.gcn_label = np.array(self.gcn_label)

    # 对某一类别的样本计算edge_weight和label
    def compute_subj_dti_edge_weight(self, main_subj_dir, suj_label):
        subj_dir = os.listdir(main_subj_dir)
        # 对该类中每个样本计算特征
        for i in range(len(subj_dir)):
            subj_file_name = main_subj_dir + '/' + subj_dir[i] + '/RegionSeries' + self.atlas + '.mat'
            region_series = scipy.io.loadmat(subj_file_name)['RegionSeries']
            subj_file_name = main_subj_dir + '/' + subj_dir[i] + '/' + self.dti_file_name + self.atlas + '.mat'
            dti_edge_weight = scipy.io.loadmat(subj_file_name)['connectivity']
            i_subj_dti_edge_weight = np.zeros((self.adj_number, self.node_number, self.node_number))
            i_subj_node_label_signal = np.zeros((self.adj_number, self.node_number, self.window_size))
            # 对该样本每个时间窗计算edge weight
            for j in range(0, self.total_window_size - self.window_size, self.step):
                sub_region_series = region_series[j:j + self.window_size, :]
                i_subj_node_label_signal[math.ceil(j / self.step), :, :] = np.transpose(sub_region_series)
                i_subj_dti_edge_weight[math.ceil(j / self.step), :, :] = dti_edge_weight
            self.dti_edge_weight_matrix.append(i_subj_dti_edge_weight)
            self.gcn_node_label_signal_matrix.append(i_subj_node_label_signal)
            self.gcn_label.append(suj_label)

    # 计算edge_weight_matrix_binary
    def compute_dti_edge_weight_binary(self):
        self.dti_edge_weight_matrix = np.array(self.dti_edge_weight_matrix)
        self.dti_edge_weight_matrix_binary = np.zeros(self.dti_edge_weight_matrix.shape)
        # 对每个样本计算二值化邻接矩阵
        self.dti_edge_weight_matrix_binary[self.dti_edge_weight_matrix > 0] = 1

    # 计算edge_weight_matrix_binary
    def compute_dti_edge_weight_binary_proportion(self):
        self.dti_edge_weight_matrix = np.array(self.dti_edge_weight_matrix)
        self.dti_edge_weight_matrix_binary = np.zeros(self.dti_edge_weight_matrix.shape)
        # 对每个样本计算二值化邻接矩阵
        for i in range(self.dti_edge_weight_matrix.shape[0]):
            # 对每个窗口计算二值化邻接矩阵
            for j in range(self.dti_edge_weight_matrix.shape[1]):
                self.dti_edge_weight_matrix_binary[i, j, :, :] = self.compute_subj_dti_edge_weight_binary(
                    self.dti_edge_weight_matrix[i, j, :, :])

    # 使用比例对单个邻接矩阵进行二值化计算，返回二值化矩阵
    def compute_subj_dti_edge_weight_binary(self, subj_edge_weight):
        # 取矩阵上三角数据
        edge_weight_list = self.mat_to_list(subj_edge_weight)
        # 根据比例，计算需保留的个数
        reserve_num = int(round(edge_weight_list.shape[1] * self.proportion))
        # 排序
        edge_weight_list_sorted = np.sort(edge_weight_list)
        # 获取阈值
        threshold = edge_weight_list_sorted[0, -(reserve_num + 1)]
        edge_weight_binary = np.zeros((self.node_number, self.node_number))
        # 大于阈值部分置1，其余为0
        edge_weight_binary[subj_edge_weight > threshold] = 1
        # scipy.io.savemat("data.mat", {'edge_weight_binary': edge_weight_binary})
        return edge_weight_binary

    # 取矩阵上三角，并转换为一维向量
    def mat_to_list(self, graph_matrix):
        graph_list = np.zeros((1, int(self.node_number * (self.node_number - 1) / 2)))
        index_start = 0
        for i in range(self.node_number - 1):
            index_end = index_start + self.node_number - i - 1
            graph_list[0, index_start: index_end] = graph_matrix[i, (i + 1): self.node_number]
            index_start = index_end
        # graph_list[0, index_start] = graph_matrix[self.node_number - 2, self.node_number - 1]
        return graph_list


# 继承于PreProcess，做HC_MDD分类
class HCMDDPreProcess(PreProcess):
    def __init__(self, atlas='', node_number=90, window_size=100, step=1, proportion=0.2, data_dir='zhongda_data_dti'):
        super(HCMDDPreProcess, self).__init__(atlas, node_number, window_size, step, proportion, data_dir)
        self.subj_type_set = ['HC', 'RMD', 'UMD']
        self.subj_label_set = [0, 1, 1]


class XinXiangHCMDDPreProcess(PreProcess):
    def __init__(self, atlas='', node_number=90, window_size=100, step=1, proportion=0.2, data_dir='xinxiang_data_dti'):
        super(XinXiangHCMDDPreProcess, self).__init__(atlas, node_number, window_size, step, proportion, data_dir)
        self.subj_type_set = ['HC', 'MDD']
        self.subj_label_set = [0, 1]


# 继承于PreProcess，做分RD_NRD类
class RDNRDPreProcess(PreProcess):
    def __init__(self, atlas='', node_number=90, window_size=100, step=1, proportion=0.2, data_dir='zhongda_data_dti'):
        super(RDNRDPreProcess, self).__init__(atlas, node_number, window_size, step, proportion, data_dir)
        self.subj_type_set = ['RMD', 'UMD']
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
