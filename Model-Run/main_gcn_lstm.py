import graphcnn.setup.dti_fmri_pre_process as multi_pre_process
from graphcnn.experiment import *
import numpy as np
import scipy

class GCLSTMHiNetConstructor(object):
    def __init__(self):
        self.no_count = 1

    def create_network(self, net, input_data):
        net.create_network(input_data)
        net.make_graphcnn_layer(32)
        net.make_graphcnn_layer(32)
        net.make_hierarchical_pooling54()
        net.make_graphcnn_layer(16)
        net.make_hierarchical_pooling14()
        net.make_graphcnn_layer(8)
        net.make_fc_layer(2, name='final', with_bn=False, with_act_func=False)
        print('no_count:', self.no_count)
        self.no_count = self.no_count + 1


class GCLSTMCrossGraphConstructor(object):
    def __init__(self):
        self.no_count = 1

    def create_network(self, net, input_data):
        net.create_network(input_data)
        net.make_cross_graph_layer(32)
        net.make_cross_graph_layer(32)
        net.make_hierarchical_pooling54()
        net.make_cross_graph_layer(16)
        net.make_hierarchical_pooling14()
        net.make_cross_graph_layer(8)
        net.make_fc_layer(2, name='final', with_bn=False, with_act_func=False)
        print('no_count:', self.no_count)
        self.no_count = self.no_count + 1


class GCLSTMAMConstructor(object):
    def __init__(self):
        self.no_count = 1

    def create_network(self, net, input_data):
        net.create_network(input_data)
        net.make_am_gcn_layer(32, if_save=True)
        net.make_am_gcn_layer(32)
        net.make_hierarchical_pooling54()
        net.make_am_gcn_layer(16)
        net.make_hierarchical_pooling14()
        net.make_am_gcn_layer(8)
        net.make_fc_layer(2, name='final', with_bn=False, with_act_func=False)
        print('no_count:', self.no_count)
        self.no_count = self.no_count + 1


class GCLSTMAMGATConstructor(object):
    def __init__(self):
        self.no_count = 1

    def create_network(self, net, input_data):
        net.create_network(input_data)
        net.make_am_gat_layer(32, if_save=True)
        net.make_am_gat_layer(32)
        net.make_hierarchical_pooling54()
        net.make_am_gat_layer(16)
        net.make_hierarchical_pooling14()
        net.make_am_gat_layer(8)
        net.make_fc_layer(2, name='final', with_bn=False, with_act_func=False)
        print('no_count:', self.no_count)
        self.no_count = self.no_count + 1

class GCLSTMSTConstructor(object):
    def __init__(self):
        self.no_count = 1

    def create_network(self, net, input_data):
        net.create_network_st(input_data)
        net.make_graphcnn_layer(32)
        net.make_graphcnn_layer(32)
        net.make_hierarchical_pooling54()
        net.make_graphcnn_layer(16)
        net.make_hierarchical_pooling14()
        net.make_graphcnn_layer(8)
        net.make_fc_layer(2, name='final', with_bn=False, with_act_func=False)
        print('no_count:', self.no_count)
        self.no_count = self.no_count + 1


class GCLSTMNetConstructor(object):
    def __init__(self):
        self.no_count = 1

    def create_network(self, net, input_data):
        net.create_network(input_data)
        net.make_graphcnn_layer(32)
        net.make_graphcnn_layer(32)
        net.make_graph_embed_pooling(no_vertices=54)
        net.make_graphcnn_layer(16)
        net.make_graph_embed_pooling(no_vertices=14)
        net.make_graphcnn_layer(8)
        net.make_fc_layer(2, name='final', with_bn=False, with_act_func=False)
        print('no_count:', self.no_count)
        self.no_count = self.no_count + 1


def test_one_sample(proportion, atlas, node_number, window_size, step, constructor, train_iterations):
    input_data_process = multi_pre_process.HCMDDPreProcess(proportion=proportion, atlas=atlas, node_number=node_number,
                                                           window_size=window_size, step=step)
    data_set = input_data_process.compute_graph_cnn_input()
    random_s = np.array([25, 50, 100, 125, 150, 175, 200, 225, 250, 275], dtype=int)
    run_experiment(data_set, constructor, proportion, train_iterations, 'HC_MDD', random_s)


def rd_nrd_test_one_sample(proportion, atlas, node_number, window_size, step, constructor, train_iterations):
    input_data_process = multi_pre_process.RDNRDPreProcess(proportion=proportion, atlas=atlas, node_number=node_number,
                                                           window_size=window_size, step=step)
    data_set = input_data_process.compute_graph_cnn_input()
    random_s = np.array([25, 50, 100, 125, 150, 175, 200, 75, 250, 275], dtype=int)
    run_experiment(data_set, constructor, proportion, train_iterations, 'RD_NRD', random_s)


def xinxiang_test_one_sample(proportion, atlas, node_number, window_size, step, constructor, train_iterations):
    input_data_process = multi_pre_process.XinXiangHCMDDPreProcess(proportion=proportion, atlas=atlas,
                                                                   node_number=node_number,
                                                                   window_size=window_size, step=step)
    data_set = input_data_process.compute_graph_cnn_input()
    random_s = np.array([25, 75, 100, 125, 150, 175, 200, 225, 250, 300], dtype=int)
    run_experiment(data_set, constructor, proportion, train_iterations, 'XinXiang', random_s)


def two_center_test_one_sample(proportion, atlas, node_number, window_size, step, constructor, train_iterations):
    input_data_process_zhongda = multi_pre_process.HCMDDPreProcess(proportion=proportion, atlas=atlas,
                                                                   node_number=node_number,
                                                                   window_size=window_size, step=step)
    a, b, c, d, e, f, label = input_data_process_zhongda.compute_graph_cnn_input()
    input_data_process_xinxiang = multi_pre_process.XinXiangHCMDDPreProcess(proportion=proportion, atlas=atlas,
                                                                            node_number=node_number,
                                                                            window_size=window_size, step=step)
    a_x, b_x, c_x, d_x, e_x, f_x, label_x = input_data_process_xinxiang.compute_graph_cnn_input()

    data_set = [np.concatenate((a, a_x)), np.concatenate((b, b_x)), np.concatenate((c, c_x)), np.concatenate((d, d_x)),
                np.concatenate((e, e_x)), np.concatenate((f, f_x)), np.concatenate((label, label_x))]
    random_s = np.array([25, 50, 100, 125, 150, 175, 200, 225, 250, 275], dtype=int)
    run_experiment(data_set, constructor, proportion, train_iterations, 'two_Center', random_s)


def run_experiment(data_set, constructor, proportion, train_iterations, name, random_s):
    acc_set = np.zeros((itertime, 1))
    std_set = np.zeros((itertime, 1))
    sen_set = np.zeros((itertime, 1))
    sen_std_set = np.zeros((itertime, 1))
    spe_set = np.zeros((itertime, 1))
    spe_std_set = np.zeros((itertime, 1))
    attr_set= []
    for iter_num in range(itertime):
        # Decay value for BatchNorm layers, seems to work better with 0.3
        GraphCNNGlobal.BN_DECAY = 0.3

        exp = GraphCNNExperiment(name+str(iter_num), 'gcn_lstm', constructor())

        exp.num_iterations = train_iterations
        exp.train_batch_size = train_batch_size
        exp.optimizer = 'adam'
        exp.debug = True

        exp.preprocess_data(data_set)
        acc, std, mean_sensitivity, std_sensitivity, mean_specificity, std_specificity, attr = exp.run_kfold_experiments(
            no_folds=10, random_state=random_s[iter_num], iter_num=iter_num)
        print_ext('10-fold: %.2f (+- %.2f)' % (acc, std))
        print_ext('sensitivity is: %.2f (+- %.2f)' % (mean_sensitivity, std_sensitivity))
        print_ext('specificity is: %.2f (+- %.2f)' % (mean_specificity, std_specificity))

        attr_set.append(attr)
        acc_set[iter_num] = acc
        std_set[iter_num] = std
        sen_set[iter_num] = mean_sensitivity
        sen_std_set[iter_num] = std_sensitivity
        spe_set[iter_num] = mean_specificity
        spe_std_set[iter_num] = std_specificity

    attr_set = np.array(attr_set)
    path = 'results/' + name + '.mat'
    scipy.io.savemat(path, {'attr_set': attr_set})
    acc_mean = np.mean(acc_set)
    acc_std = np.std(acc_set)
    sen_mean = np.mean(sen_set)
    sen_std = np.std(sen_set)
    spe_mean = np.mean(spe_set)
    spe_std = np.std(spe_set)
    print_ext('finish!')
    verify_dir_exists('results/')
    with open('results/10-10_fold_extra.txt', 'a+') as file:
        for iter_num in range(itertime):
            print_ext('acc %d :    %.2f   sen :    %.2f   spe :    %.2f' % (
                iter_num, acc_set[iter_num], sen_set[iter_num], spe_set[iter_num]))
            file.write('%s\tacc %d :   \t%.2f (+- %.2f)\tsen :   \t%.2f (+- %.2f)\tspe :   \t%.2f (+- %.2f)\n' % (
                str(datetime.now()), iter_num, acc_set[iter_num], std_set[iter_num], sen_set[iter_num],
                sen_std_set[iter_num], spe_set[iter_num], spe_std_set[iter_num]))
        print_ext('acc:     %.2f(+-%.2f)   sen:     %.2f(+-%.2f)   spe:     %.2f(+-%.2f)' % (
            acc_mean, acc_std, sen_mean, sen_std, spe_mean, spe_std))
        file.write('%s\t %.2f acc  :   \t%.2f (+- %.2f)  sen  :   \t%.2f (+- %.2f)  spe  :   \t%.2f (+- %.2f)\n' % (
            str(datetime.now()), proportion, acc_mean, acc_std, sen_mean, sen_std, spe_mean, spe_std))


def main_gcn_lstm():
    proportion = 0.20
    atlas = ''  # 改
    node_number = 90  # 改
    constructor = GCLSTMAMConstructor  # GCLSTMCrossGraphConstructor  # GCLSTMSTConstructor  # GCLSTMHiNetConstructor
    window_size = 100
    step = 2
    train_iterations = 800

    flag = tf.app.flags
    flag.DEFINE_integer('adj_num', 66, 'adj_num')  # 66
    flag.DEFINE_integer('node_number', node_number, 'node_number')

    constructor = GCLSTMAMConstructor
    # for proportion in np.arange(0.02, 0.32, 0.02):
    #     ## 中大
    #     # test_one_sample(proportion, atlas, node_number, window_size, step, constructor, train_iterations)
        
    #     ## 新乡
    #     # xinxiang_test_one_sample(proportion, atlas, node_number, window_size, step, constructor, train_iterations)
        
    #     ## two-site
    #     # two_center_test_one_sample(proportion, atlas, node_number, window_size, step, constructor, train_iterations)
    
    # for window_size in np.arange(60, 130, 10):
    #     ## 中大
    #     # test_one_sample(proportion, atlas, node_number, window_size, step, constructor, train_iterations)
        
    #     ## 新乡
    #     # xinxiang_test_one_sample(proportion, atlas, node_number, window_size, step, constructor, train_iterations)
        
    #     ## two-site
    #     # two_center_test_one_sample(proportion, atlas, node_number, window_size, step, constructor, train_iterations)
        
    #rd_nrd_test_one_sample(proportion, atlas, node_number, window_size, step, constructor, train_iterations)


train_batch_size = 30
itertime = 10
main_gcn_lstm()
