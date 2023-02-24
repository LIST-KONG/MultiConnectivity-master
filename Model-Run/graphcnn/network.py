from graphcnn.layers import *
import tensorflow as tf2
import numpy as np
from tensorflow.python.ops.rnn import dynamic_rnn
import graphcnn.setup.dti_pre_process as pre_process
import graphcnn.gc_lstm as gc_lstm
import graphcnn.gat_lstm as gat_lstm

no_ite = 0
FLAGS = tf.app.flags.FLAGS


class GraphCNNNetwork(object):
    def __init__(self):
        self.fMRI_V = None
        self.fMRI_A = None
        self.DTI_V = None
        self.DTI_A = None
        self.dynamic_fMRI_V = None
        self.dynamic_fMRI_A = None
        self.labels = None
        self.result = None
        self.network_debug = False
        self.pooling_weight54 = None
        self.pooling_weight14 = None
        self.loss = 0
        self.attr = None

    def create_network(self, input_data):
        self.fMRI_V = input_data[0]
        self.fMRI_A = input_data[1]
        self.DTI_V = input_data[2]
        self.DTI_A = input_data[3]
        self.dynamic_fMRI_V = input_data[4]
        self.dynamic_fMRI_A = input_data[5]
        self.labels = input_data[6]
        self.dynamic_fMRI_V = tf.reshape(self.dynamic_fMRI_V,
                                         [-1, FLAGS.adj_num, FLAGS.node_number, self.dynamic_fMRI_V.get_shape()[3]])
        # self.dynamic_fMRI_V = tf.Print(self.dynamic_fMRI_V, [tf.shape(self.dynamic_fMRI_V)])
        self.dynamic_fMRI_A = tf.reshape(self.dynamic_fMRI_A,
                                         [-1, FLAGS.adj_num, FLAGS.node_number, self.dynamic_fMRI_A.get_shape()[3],
                                          FLAGS.node_number])
        pooling_weight54, pooling_weight14 = pre_process.compute_pooling_weight()
        self.pooling_weight54 = tf.constant(pooling_weight54, dtype=tf.float32)
        self.pooling_weight14 = tf.constant(pooling_weight14, dtype=tf.float32)
        self.loss = 0
        return input_data

    def create_network_st(self, input_data):
        self.dynamic_fMRI_V = input_data[4]
        self.dynamic_fMRI_A = input_data[5]
        self.labels = input_data[6]
        self.dynamic_fMRI_V = tf.reshape(self.dynamic_fMRI_V,
                                         [-1, FLAGS.adj_num, FLAGS.node_number, self.dynamic_fMRI_V.get_shape()[3]])
        # self.dynamic_fMRI_V = tf.Print(self.dynamic_fMRI_V, [tf.shape(self.dynamic_fMRI_V)])
        self.dynamic_fMRI_A = tf.reshape(self.dynamic_fMRI_A,
                                         [-1, FLAGS.adj_num, FLAGS.node_number, self.dynamic_fMRI_A.get_shape()[3],
                                          FLAGS.node_number])
        pooling_weight54, pooling_weight14 = pre_process.compute_pooling_weight()
        self.pooling_weight54 = tf.constant(pooling_weight54, dtype=tf.float32)
        self.pooling_weight14 = tf.constant(pooling_weight14, dtype=tf.float32)
        return input_data

    def make_batchnorm_layer(self):
        self.fMRI_V = make_bn(self.fMRI_V, self.is_training, num_updates=self.global_step)
        self.DTI_V = make_bn(self.DTI_V, self.is_training, num_updates=self.global_step)
        self.dynamic_fMRI_V = make_bn(self.dynamic_fMRI_V, self.is_training, num_updates=self.global_step)
        return self.fMRI_V, self.DTI_V, self.dynamic_fMRI_V

    # Equivalent to 0-hop filter
    def make_embedding_layer(self, no_filters, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='Embed') as scope:
            self.fMRI_V = make_embedding_layer(self.fMRI_V, no_filters)
            self.DTI_V = make_embedding_layer(self.DTI_V, no_filters)
            # self.dynamic_fMRI_V = make_embedding_layer(self.dynamic_fMRI_V, no_filters)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                if self.fMRI_V is not None:
                    self.fMRI_V = tf.nn.relu(self.fMRI_V)
                if self.DTI_V is not None:
                    self.DTI_V = tf.nn.relu(self.DTI_V)
                if self.dynamic_fMRI_V is not None:
                    self.dynamic_fMRI_V = tf.nn.relu(self.dynamic_fMRI_V)
        return self.fMRI_V, self.fMRI_A, self.DTI_V, self.DTI_A, self.dynamic_fMRI_V, self.dynamic_fMRI_A

    def make_dropout_layer(self, keep_prob=0.5):
        self.fMRI_V = tf.cond(self.is_training, lambda: tf.nn.dropout(self.fMRI_V, keep_prob=keep_prob),
                              lambda: self.fMRI_V)
        self.DTI_V = tf.cond(self.is_training, lambda: tf.nn.dropout(self.DTI_V, keep_prob=keep_prob),
                             lambda: self.DTI_V)
        self.dynamic_fMRI_V = tf.cond(self.is_training, lambda: tf.nn.dropout(self.dynamic_fMRI_V, keep_prob=keep_prob),
                                      lambda: self.dynamic_fMRI_V)
        return self.fMRI_V, self.DTI_V, self.dynamic_fMRI_V

    def make_graphcnn_layer(self, no_filters, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='Graph-CNN') as scope:
            self.fMRI_V = make_graphcnn_layer(self.fMRI_V, self.fMRI_A, no_filters)
            self.DTI_V = make_graphcnn_layer(self.DTI_V, self.DTI_A, no_filters)
            attention_V = make_embedding_layer(tf.concat([self.fMRI_V, self.DTI_V], axis=2), 1)
            # attention_V = tf.Print(attention_V, [attention_V[0, :, :]], summarize=256)
            attention_V = tf.nn.softmax(attention_V, axis=1)
            # attention_V = tf.reduce_mean(make_embedding_layer(tf.concat([self.fMRI_V, self.DTI_V], axis=2), 1), axis=0)
            # attention_V = tf.nn.softmax(
            #     tf.reduce_mean(make_embedding_layer(tf.concat([self.fMRI_V, self.DTI_V], axis=2), 1), axis=0), axis=0)
            # attention_V = tf.Print(attention_V, [attention_V[0, :, :]], summarize=256)
            self.dynamic_fMRI_V = gc_lstm.gcnlstm_loop(lstm_size=FLAGS.adj_num, input_data_V=self.dynamic_fMRI_V,
                                                       input_data_A=self.dynamic_fMRI_A, no_filter=no_filters,
                                                       attention_V=attention_V, if_concat=False)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                if self.fMRI_V is not None:
                    self.fMRI_V = tf.nn.relu(self.fMRI_V)
                if self.DTI_V is not None:
                    self.DTI_V = tf.nn.relu(self.DTI_V)
                if self.dynamic_fMRI_V is not None:
                    self.dynamic_fMRI_V = tf.nn.relu(self.dynamic_fMRI_V)
        return self.fMRI_V, self.DTI_V, self.dynamic_fMRI_V

    def make_am_gcn_layer(self, no_filters, name=None, with_bn=True, with_act_func=True, if_save=False):
        with tf.variable_scope(name, default_name='Graph-CNN') as scope:
            fmri_self = make_graphcnn_layer(self.fMRI_V, self.fMRI_A, no_filters, name=None)
            dti_self = make_graphcnn_layer(self.DTI_V, self.DTI_A, no_filters, name=None)
            fmri_share, dti_share = make_am_gcn_layer(self.fMRI_V, self.fMRI_A, self.DTI_V, self.DTI_A, no_filters,
                                                      name=None)
            self.fMRI_V = (fmri_share + fmri_self)/2
            self.DTI_V = (dti_share + dti_self)/2

            self.loss = self.loss + tf.nn.l2_loss(fmri_share-dti_share)

            attention_V = make_embedding_layer(tf.concat([self.fMRI_V, self.DTI_V], axis=2), 1)
            # attention_V = tf.reduce_mean(make_embedding_layer(tf.concat([self.fMRI_V, self.DTI_V], axis=2), 1), axis=0)
            # attention_V = tf.nn.softmax(
            #     tf.reduce_mean(make_embedding_layer(tf.concat([self.fMRI_V, self.DTI_V], axis=2), 1), axis=0), axis=0)
            self.dynamic_fMRI_V = gc_lstm.gcnlstm_loop(lstm_size=FLAGS.adj_num, input_data_V=self.dynamic_fMRI_V,
                                                       input_data_A=self.dynamic_fMRI_A, no_filter=no_filters,
                                                       attention_V=attention_V, if_concat=False)

            if if_save:
                self.attr = attention_V
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                if self.fMRI_V is not None:
                    self.fMRI_V = tf.nn.relu(self.fMRI_V)
                if self.DTI_V is not None:
                    self.DTI_V = tf.nn.relu(self.DTI_V)
                if self.dynamic_fMRI_V is not None:
                    self.dynamic_fMRI_V = tf.nn.relu(self.dynamic_fMRI_V)
        return self.fMRI_V, self.DTI_V, self.dynamic_fMRI_V
    
    def make_am_gat_layer(self, no_filters, name=None, with_bn=True, with_act_func=True, if_save=False):
        with tf.variable_scope(name, default_name='Graph-Attention') as scope:
            fmri_self = make_graphattention_layer(self.fMRI_V, self.fMRI_A, no_filters, name=None)
            dti_self = make_graphattention_layer(self.DTI_V, self.DTI_A, no_filters, name=None)
            fmri_share, dti_share = make_am_gat_layer(self.fMRI_V, self.fMRI_A, self.DTI_V, self.DTI_A, no_filters,
                                                      name=None)
            self.fMRI_V = (fmri_share + fmri_self)/2
            self.DTI_V = (dti_share + dti_self)/2

            self.loss = self.loss + tf.nn.l2_loss(fmri_share-dti_share)

            attention_V = make_embedding_layer(tf.concat([self.fMRI_V, self.DTI_V], axis=2), 1)
            # attention_V = tf.reduce_mean(make_embedding_layer(tf.concat([self.fMRI_V, self.DTI_V], axis=2), 1), axis=0)
            # attention_V = tf.nn.softmax(
            #     tf.reduce_mean(make_embedding_layer(tf.concat([self.fMRI_V, self.DTI_V], axis=2), 1), axis=0), axis=0)
            self.dynamic_fMRI_V = gc_lstm.gcnlstm_loop(lstm_size=FLAGS.adj_num, input_data_V=self.dynamic_fMRI_V,
                                                       input_data_A=self.dynamic_fMRI_A, no_filter=no_filters,
                                                       attention_V=attention_V, if_concat=False)

            if if_save:
                self.attr = attention_V
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                if self.fMRI_V is not None:
                    self.fMRI_V = tf.nn.relu(self.fMRI_V)
                if self.DTI_V is not None:
                    self.DTI_V = tf.nn.relu(self.DTI_V)
                if self.dynamic_fMRI_V is not None:
                    self.dynamic_fMRI_V = tf.nn.relu(self.dynamic_fMRI_V)
        return self.fMRI_V, self.DTI_V, self.dynamic_fMRI_V
    
    def make_am_gin_layer(self, no_filters, name=None, with_bn=True, with_act_func=True, if_save=False):
        with tf.variable_scope(name, default_name='Graph-CNN') as scope:
            fmri_self = make_graphattention_layer(self.fMRI_V, self.fMRI_A, no_filters, name=None)
            dti_self = make_graphattention_layer(self.DTI_V, self.DTI_A, no_filters, name=None)
            fmri_share, dti_share = make_am_gat_layer(self.fMRI_V, self.fMRI_A, self.DTI_V, self.DTI_A, no_filters,
                                                      name=None)
            self.fMRI_V = (fmri_share + fmri_self)/2
            self.DTI_V = (dti_share + dti_self)/2

            self.loss = self.loss + tf.nn.l2_loss(fmri_share-dti_share)

            attention_V = make_embedding_layer(tf.concat([self.fMRI_V, self.DTI_V], axis=2), 1)
            # attention_V = tf.reduce_mean(make_embedding_layer(tf.concat([self.fMRI_V, self.DTI_V], axis=2), 1), axis=0)
            # attention_V = tf.nn.softmax(
            #     tf.reduce_mean(make_embedding_layer(tf.concat([self.fMRI_V, self.DTI_V], axis=2), 1), axis=0), axis=0)
            self.dynamic_fMRI_V = gc_lstm.gcnlstm_loop(lstm_size=FLAGS.adj_num, input_data_V=self.dynamic_fMRI_V,
                                                       input_data_A=self.dynamic_fMRI_A, no_filter=no_filters,
                                                       attention_V=attention_V, if_concat=False)

            if if_save:
                self.attr = attention_V
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                if self.fMRI_V is not None:
                    self.fMRI_V = tf.nn.relu(self.fMRI_V)
                if self.DTI_V is not None:
                    self.DTI_V = tf.nn.relu(self.DTI_V)
                if self.dynamic_fMRI_V is not None:
                    self.dynamic_fMRI_V = tf.nn.relu(self.dynamic_fMRI_V)
        return self.fMRI_V, self.DTI_V, self.dynamic_fMRI_V

    def make_cross_graph_layer(self, no_filters, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='Graph-CNN') as scope:
            self.fMRI_V, self.DTI_V = make_cross_graph_layer(self.fMRI_V, self.fMRI_A, self.DTI_V, self.DTI_A,
                                                             no_filters)
            attention_V = make_embedding_layer(tf.concat([self.fMRI_V, self.DTI_V], axis=2), 1)
            # attention_V = tf.reduce_mean(make_embedding_layer(tf.concat([self.fMRI_V, self.DTI_V], axis=2), 1), axis=0)
            # attention_V = tf.nn.softmax(
            #     tf.reduce_mean(make_embedding_layer(tf.concat([self.fMRI_V, self.DTI_V], axis=2), 1), axis=0), axis=0)
            self.dynamic_fMRI_V = gc_lstm.gcnlstm_loop(lstm_size=FLAGS.adj_num, input_data_V=self.dynamic_fMRI_V,
                                                       input_data_A=self.dynamic_fMRI_A, no_filter=no_filters,
                                                       attention_V=attention_V, if_concat=False)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                if self.fMRI_V is not None:
                    self.fMRI_V = tf.nn.relu(self.fMRI_V)
                if self.DTI_V is not None:
                    self.DTI_V = tf.nn.relu(self.DTI_V)
                if self.dynamic_fMRI_V is not None:
                    self.dynamic_fMRI_V = tf.nn.relu(self.dynamic_fMRI_V)
        return self.fMRI_V, self.DTI_V, self.dynamic_fMRI_V

    def make_graph_embed_pooling(self, no_vertices=1, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='GraphEmbedPool') as scope:
            self.fMRI_V, self.fMRI_A = make_graph_embed_pooling(self.fMRI_V, self.fMRI_A, no_vertices=no_vertices)
            self.DTI_V, self.DTI_A = make_graph_embed_pooling(self.DTI_V, self.DTI_A, no_vertices=no_vertices)
            V_shape = self.dynamic_fMRI_V.get_shape()
            A_shape = self.dynamic_fMRI_A.get_shape()
            reshape_V = tf.reshape(self.dynamic_fMRI_V, (-1, V_shape[2], V_shape[3]))
            reshape_A = tf.reshape(self.dynamic_fMRI_A, (-1, A_shape[2], A_shape[3], A_shape[4]))
            self.dynamic_fMRI_V, self.dynamic_fMRI_A = make_graph_embed_pooling(reshape_V, reshape_A,
                                                                                no_vertices=no_vertices)
            self.dynamic_fMRI_V = tf.reshape(self.dynamic_fMRI_V, (-1, V_shape[1], no_vertices, V_shape[3]))
            self.dynamic_fMRI_A = tf.reshape(self.dynamic_fMRI_A,
                                             (-1, A_shape[1], no_vertices, A_shape[3], no_vertices))

            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                if self.fMRI_V is not None:
                    self.fMRI_V = tf.nn.relu(self.fMRI_V)
                if self.DTI_V is not None:
                    self.DTI_V = tf.nn.relu(self.DTI_V)
                if self.dynamic_fMRI_V is not None:
                    self.dynamic_fMRI_V = tf.nn.relu(self.dynamic_fMRI_V)
        return self.fMRI_V, self.fMRI_A, self.DTI_V, self.DTI_A, self.dynamic_fMRI_V, self.dynamic_fMRI_A

    def make_hierarchical_pooling54(self, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='HierarchicalPool') as scope:
            self.fMRI_V, self.fMRI_A = make_hierarchical_pooling54(self.fMRI_V, self.fMRI_A,
                                                                   factors=self.pooling_weight54)
            self.DTI_V, self.DTI_A = make_hierarchical_pooling54(self.DTI_V, self.DTI_A,
                                                                 factors=self.pooling_weight54)
            self.dynamic_fMRI_V, self.dynamic_fMRI_A = make_st_pooling54(self.dynamic_fMRI_V, self.dynamic_fMRI_A,
                                                                         factors=self.pooling_weight54)

            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                if self.fMRI_V is not None:
                    self.fMRI_V = tf.nn.relu(self.fMRI_V)
                if self.DTI_V is not None:
                    self.DTI_V = tf.nn.relu(self.DTI_V)
                if self.dynamic_fMRI_V is not None:
                    self.dynamic_fMRI_V = tf.nn.relu(self.dynamic_fMRI_V)
        return self.fMRI_V, self.fMRI_A, self.DTI_V, self.DTI_A, self.dynamic_fMRI_V, self.dynamic_fMRI_A

    def make_hierarchical_pooling14(self, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='HierarchicalPool') as scope:
            self.fMRI_V, self.fMRI_A = make_hierarchical_pooling14(self.fMRI_V, self.fMRI_A,
                                                                   factors=self.pooling_weight14)
            self.DTI_V, self.DTI_A = make_hierarchical_pooling14(self.DTI_V, self.DTI_A,
                                                                 factors=self.pooling_weight14)
            self.dynamic_fMRI_V, self.dynamic_fMRI_A = make_st_pooling14(self.dynamic_fMRI_V, self.dynamic_fMRI_A,
                                                                         factors=self.pooling_weight14)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                if self.fMRI_V is not None:
                    self.fMRI_V = tf.nn.relu(self.fMRI_V)
                if self.DTI_V is not None:
                    self.DTI_V = tf.nn.relu(self.DTI_V)
                if self.dynamic_fMRI_V is not None:
                    self.dynamic_fMRI_V = tf.nn.relu(self.dynamic_fMRI_V)
        return self.fMRI_V, self.fMRI_A, self.DTI_V, self.DTI_A, self.dynamic_fMRI_V, self.dynamic_fMRI_A

    def make_fc_layer(self, no_filters, name=None, with_bn=False, with_act_func=True):
        with tf.variable_scope(name, default_name='FC') as scope:
            self.result = None
            if self.fMRI_V is not None and len(self.fMRI_V.get_shape()) > 2:
                no_input_features = int(np.prod(self.fMRI_V.get_shape()[1:]))
                self.fMRI_V = tf.reshape(self.fMRI_V, [-1, no_input_features])
                self.result = self.fMRI_V
            if self.DTI_V is not None and len(self.DTI_V.get_shape()) > 2:
                no_input_features = int(np.prod(self.DTI_V.get_shape()[1:]))
                self.DTI_V = tf.reshape(self.DTI_V, [-1, no_input_features])
                if self.result is not None:
                    self.result = tf.concat([self.result, self.DTI_V], 1)
                else:
                    self.result = self.DTI_V
            if self.dynamic_fMRI_V is not None and len(self.dynamic_fMRI_V.get_shape()) > 2:
                self.dynamic_fMRI_V = tf.reduce_mean(self.dynamic_fMRI_V, axis=1)
                no_input_features = int(np.prod(self.dynamic_fMRI_V.get_shape()[1:]))
                self.dynamic_fMRI_V = tf.reshape(self.dynamic_fMRI_V, [-1, no_input_features])
                if self.result is not None:
                    self.result = tf.concat([self.result, self.dynamic_fMRI_V], 1)
                else:
                    self.result = self.dynamic_fMRI_V
            self.result = make_embedding_layer(self.result, no_filters)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.result = tf.nn.relu(self.result)
        return self.result
