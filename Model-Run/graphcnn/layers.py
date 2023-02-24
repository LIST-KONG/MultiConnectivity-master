from unittest import result
from graphcnn.helper import *
import tensorflow as tf
import math
from tensorflow.contrib.layers.python.layers import utils
import numpy as np


FLAGS = tf.app.flags.FLAGS


def make_variable(name, shape, initializer=tf.truncated_normal_initializer(), regularizer=None):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer, dtype=dtype)
    return var


def make_bias_variable(name, shape):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1), dtype=dtype)
    return var


def make_variable_with_weight_decay(name, shape, stddev=0.01, wd=0.0005):
    dtype = tf.float32
    regularizer = None
    if wd is not None and wd > 1e-7:
        def regularizer(var):
            return tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    var = make_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev),
                        regularizer=regularizer)
    return var


def make_attention_variable_with_weight_decay(name, shape, wd=0.0005):
    dtype = tf.float32
    regularizer = None
    if wd is not None and wd > 1e-7:
        def regularizer(var):
            return tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    var = make_variable(name, shape, initializer=tf.zeros_initializer(), regularizer=regularizer)
    return var


def make_bn(input, phase, axis=-1, epsilon=0.001, mask=None, num_updates=None, name=None):
    default_decay = GraphCNNGlobal.BN_DECAY
    with tf.variable_scope(name, default_name='BatchNorm') as scope:
        if input is None:
            return input
        input_size = input.get_shape()[axis].value
        if axis == -1:
            axis = len(input.get_shape()) - 1
        axis_arr = [i for i in range(len(input.get_shape())) if i != axis]
        if mask is None:
            batch_mean, batch_var = tf.nn.moments(input, axis_arr)
        else:
            mask = tf.Print(mask, [tf.shape(mask)], message="current_mask is the size:", summarize=4)
            batch_mean, batch_var = tf.nn.weighted_moments(input, axis_arr, mask)
        gamma = make_variable('gamma', input_size, initializer=tf.constant_initializer(1))
        beta = make_bias_variable('bias', input_size)
        ema = tf.train.ExponentialMovingAverage(decay=default_decay, num_updates=num_updates)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))

        return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)


def batch_mat_mult(A, B):
    A_shape = tf.shape(A)
    A_reshape = tf.reshape(A, [-1, A_shape[-1]])

    # So the Tensor has known dimensions
    if B.get_shape()[1] is None:
        axis_2 = -1
    else:
        axis_2 = B.get_shape()[1]
    result = tf.matmul(A_reshape, B)
    result = tf.reshape(result, tf.stack([-1, A_shape[1], axis_2]))
    return result


def batch_matmul(A, B):
    A_reshape = tf.transpose(A, [0, 2, 1])
    A_reshape = tf.reshape(A_reshape, (-1, A.get_shape()[1].value))
    result = tf.matmul(A_reshape, B)
    result = tf.reshape(result, (-1, tf.shape(A)[2], B.get_shape()[1].value))
    result = tf.transpose(result, [0, 2, 1])
    return result


def make_softmax_layer(V, axis=1, name=None):
    with tf.variable_scope(name, default_name='Softmax') as scope:
        max_value = tf.reduce_max(V, axis=axis, keep_dims=True)
        exp = tf.exp(tf.subtract(V, max_value))
        prob = tf.div(exp, tf.reduce_sum(exp, axis=axis, keep_dims=True))
        return prob


def make_graphcnn_layer(V, A, no_filters, name=None):
    with tf.variable_scope(name, default_name='Graph-CNN') as scope:
        if V is None or A is None:
            return V
        no_A = A.get_shape()[2].value
        no_features = V.get_shape()[2].value
        no_V = V.get_shape()[1].value
        W = make_variable_with_weight_decay('weights', [no_features * no_A, no_filters], stddev=math.sqrt(
            1.0 / (no_features * (no_A + 1) * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        W_I = make_variable_with_weight_decay('weights_I', [no_features, no_filters], stddev=math.sqrt(
            GraphCNNGlobal.GRAPHCNN_I_FACTOR / (no_features * (no_A + 1) * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        b = make_bias_variable('bias', [no_filters])

        # 加入节点attention
        # if no_V is None:
        #     no_V = FLAGS.node_number
        # attention_V = make_attention_variable_with_weight_decay('attention_V', [1, no_V])
        # # attention_V = tf.Print(attention_V, [attention_V], message='attentionV shape:', summarize=2880)
        # V = tf.add(V, tf.transpose(tf.multiply(tf.transpose(V), tf.transpose(attention_V))))

        A_shape = tf.shape(A)
        A_reshape = tf.reshape(A, [-1, A_shape[1] * no_A, A_shape[1]])
        n = tf.matmul(A_reshape, V)
        n = tf.reshape(n, [-1, A_shape[1], no_A * no_features])
        result = batch_mat_mult(n, W) + batch_mat_mult(V, W_I) + b
        result = tf.reshape(result, [-1, no_V, no_filters])
        # result = tf.reshape(result,[V_shape[0],V_shape[1],V_shape[2],no_filters])
        return result

def make_cross_graph_layer(V1, A1, V2, A2, no_filters, name=None):
    with tf.variable_scope(name, default_name='Graph-CNN') as scope:
        if V1 is None or A1 is None:
            return V1, V2
        no_A1 = A1.get_shape()[2].value
        no_features1 = V1.get_shape()[2].value
        no_A2 = A1.get_shape()[2].value
        no_features2 = V2.get_shape()[2].value
        no_V = V2.get_shape()[1].value
        W1 = make_variable_with_weight_decay('weights1', [no_features1, no_filters], stddev=math.sqrt(
            1.0 / (no_features1 * (no_A1 + 1) * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        W2 = make_variable_with_weight_decay('weights2', [no_features2, no_filters], stddev=math.sqrt(
            1.0 / (no_features2 * (no_A2 + 1) * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))

        new_V1 = make_graphcnn_layer(V1, A1, no_filters, name=None)
        new_V2 = make_graphcnn_layer(V2, A2, no_filters, name=None)
        new_V1 = new_V1 + batch_mat_mult(V1, W1)
        new_V2 = new_V2 + batch_mat_mult(V2, W2)

        new_V1 = tf.reshape(new_V1, [-1, no_V, no_filters])
        new_V2 = tf.reshape(new_V2, [-1, no_V, no_filters])
        return new_V1, new_V2

def make_am_gcn_layer(V1, A1, V2, A2, no_filters, name=None):
    with tf.variable_scope(name, default_name='Graph-CNN') as scope:
        if V1 is None or A1 is None:
            return V1, V2
        no_A = A1.get_shape()[2].value
        no_features1 = V1.get_shape()[2].value
        no_features2 = V2.get_shape()[2].value
        no_V = V2.get_shape()[1].value

        if no_features1!=no_features2:
            W1 = make_variable_with_weight_decay('weights1', [no_features1 * no_A, no_filters], stddev=math.sqrt(
               1.0 / (no_features1 * (no_A + 1) * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
            W2 = make_variable_with_weight_decay('weights2', [no_features2 * no_A, no_filters], stddev=math.sqrt(
               1.0 / (no_features1 * (no_A + 1) * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
            V1 = batch_mat_mult(V1, W1)
            V2 = batch_mat_mult(V2, W2)

        no_features = V1.get_shape()[2].value
        W = make_variable_with_weight_decay('weights', [no_features * no_A, no_filters], stddev=math.sqrt(
            1.0 / (no_features * (no_A + 1) * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        W_I = make_variable_with_weight_decay('weights_I', [no_features, no_filters], stddev=math.sqrt(
            GraphCNNGlobal.GRAPHCNN_I_FACTOR / (no_features * (no_A + 1) * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        b = make_bias_variable('bias', [no_filters])

        A1_shape = tf.shape(A1)
        A1_reshape = tf.reshape(A1, [-1, A1_shape[1] * no_A, A1_shape[1]])
        n1 = tf.matmul(A1_reshape, V1)
        n1 = tf.reshape(n1, [-1, A1_shape[1], no_A * no_features])
        result1 = batch_mat_mult(n1, W) + batch_mat_mult(V1, W_I) + b
        result1 = tf.reshape(result1, [-1, no_V, no_filters])

        A2_shape = tf.shape(A2)
        A2_reshape = tf.reshape(A2, [-1, A2_shape[1] * no_A, A2_shape[1]])
        n2 = tf.matmul(A2_reshape, V2)
        n2 = tf.reshape(n2, [-1, A2_shape[1], no_A * no_features])
        result2 = batch_mat_mult(n2, W) + batch_mat_mult(V2, W_I) + b
        result2 = tf.reshape(result2, [-1, no_V, no_filters])

        return result1, result2

def make_am_gat_layer(V1, A1, V2, A2, no_filters, name=None):
    with tf.variable_scope(name, default_name='GAT') as scope:
        attns_1 = []
        attns_2 = []
        n_heads = [8, 1]
        hid_units = [8]
        activation=tf.nn.elu
        ffd_drop=0.0#0.6
        attn_drop=0.0#0.6
        residual = False

        no_A = A1.get_shape()[2].value
        no_features1 = V1.get_shape()[2].value
        no_features2 = V2.get_shape()[2].value
        no_V = V2.get_shape()[1].value

        if no_features1!=no_features2:
            W1 = make_variable_with_weight_decay('weights1', [no_features1 * no_A, no_filters], stddev=math.sqrt(
               1.0 / (no_features1 * (no_A + 1) * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
            W2 = make_variable_with_weight_decay('weights2', [no_features2 * no_A, no_filters], stddev=math.sqrt(
               1.0 / (no_features1 * (no_A + 1) * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))

            V1 = batch_mat_mult(V1, W1)
            V2 = batch_mat_mult(V2, W2)
        ####
        V_shape = V1.get_shape()

        A_shape = A1.get_shape()
        A1 = tf.reshape(A1, (-1, A_shape[1], A_shape[3]))
        A2 = tf.reshape(A2, (-1, A_shape[1], A_shape[3]))
        # if A_shape.rank==5:
        #     self.current_A = tf.reshape(self.current_A, (-1, A_shape[2], A_shape[3], A_shape[4]))
        #     self.current_A=tf.reshape(self.current_A, (-1, FLAGS.node_number, FLAGS.node_number))
        ####
        
        for head_num in range(n_heads[0]):
            res_1, res_2 = attn_head_shared(V1, V2, bias_mat_1=A1, bias_mat_2=A2, seed=head_num,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False, name=name)
            attns_1.append(res_1)
            attns_2.append(res_2)
        h_1 = tf.concat(attns_1, axis=-1)
        h_2 = tf.concat(attns_2, axis=-1)
        # for i in range(1, len(hid_units)):
        #     h_old = h_1
        #     attns = []
        #     for _ in range(n_heads[i]):
        #         attns.append(attn_head(h_1, bias_mat=A,
        #             out_sz=hid_units[i], activation=activation,
        #             in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
        #     h_1 = tf.concat(attns, axis=-1)
        out_1 = []
        out_2 = []
        for i in range(n_heads[-1]):
            res_1, res_2 = attn_head_shared(h_1, h_2, bias_mat_1=A1, bias_mat_2=A2, seed=i,
                out_sz=no_filters, activation=lambda x: x,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False, name=name)
            out_1.append(res_1)
            out_2.append(res_2)
        
        out_1 = tf.convert_to_tensor(out_1)
        out_2 = tf.convert_to_tensor(out_2)
        logits_1 = out_1 / n_heads[-1]
        logits_2 = out_2 / n_heads[-1]

        result_1 = tf.reshape(logits_1,(-1, V_shape[1], no_filters))
        result_2 = tf.reshape(logits_2,(-1, V_shape[1], no_filters))
    return result_1, result_2

def make_graphattention_layer(V, A, no_filters, name=None):
    with tf.variable_scope(name, default_name='GAT') as scope:
        attns = []
        n_heads = [8, 1]
        hid_units = [8]
        activation=tf.nn.elu
        ffd_drop=0.0#0.6
        attn_drop=0.0#0.6
        residual = False
        ####
        V_shape = V.get_shape()
        V = tf.reshape(V, (-1, V_shape[1], V_shape[2]))
        A_shape = A.get_shape()
        A = tf.reshape(A, (-1, A_shape[1], A_shape[3]))
        # if A_shape.rank==5:
        #     self.current_A = tf.reshape(self.current_A, (-1, A_shape[2], A_shape[3], A_shape[4]))
        #     self.current_A=tf.reshape(self.current_A, (-1, FLAGS.node_number, FLAGS.node_number))
        ####
        for _ in range(n_heads[0]):
            attns.append(attn_head(V, bias_mat=A,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False, name=name))
        h_1 = tf.concat(attns, axis=-1)
        # for i in range(1, len(hid_units)):
        #     h_old = h_1
        #     attns = []
        #     for _ in range(n_heads[i]):
        #         attns.append(attn_head(h_1, bias_mat=A,
        #             out_sz=hid_units[i], activation=activation,
        #             in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
        #     h_1 = tf.concat(attns, axis=-1)
        out = []
        for i in range(n_heads[-1]):
            out.append(attn_head(h_1, bias_mat=A,
                out_sz=no_filters, activation=lambda x: x,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False, name=name))
        
        out = tf.convert_to_tensor(out)
        logits = out / n_heads[-1]

        result = tf.reshape(logits,(-1, V_shape[1], no_filters))

    return result

def make_graph_embed_pooling(V, A, no_vertices=1, name=None):
    with tf.variable_scope(name, default_name='GraphEmbedPooling') as scope:
        factors = make_embedding_layer(V, no_vertices, name='Factors')

        factors = make_softmax_layer(factors)
        # factors = tf.Print(factors,[factors[0,0,:]],message="factor[0]: ")
        # factors = tf.Print(factors, [factors[61,0,:]], message="factor[1]: ")

        result = tf.matmul(factors, V, transpose_a=True)

        if no_vertices == 1:
            no_features = V.get_shape()[2].value
            return tf.reshape(result, [-1, no_features]), A

        result_A = tf.reshape(A, (tf.shape(A)[0], -1, tf.shape(A)[-1]))
        result_A = tf.matmul(result_A, factors)
        result_A = tf.reshape(result_A, (tf.shape(A)[0], tf.shape(A)[-1], -1))
        result_A = tf.matmul(factors, result_A, transpose_a=True)
        result_A = tf.reshape(result_A, (tf.shape(A)[0], no_vertices, A.get_shape()[2].value, no_vertices))

        # result = tf.reshape(result,[V_shape[0],V_shape[1],no_vertices,V_shape[3]])
        # result_A = tf.reshape(result_A, [A_shape[0], A_shape[1], no_vertices, A_shape[3], no_vertices])
        return result, result_A


def make_embedding_layer(V, no_filters, name=None):
    with tf.variable_scope(name, default_name='Embed') as scope:
        no_features = V.get_shape()[-1].value
        W = make_variable_with_weight_decay('weights', [no_features, no_filters], stddev=1.0 / math.sqrt(no_features))
        b = make_bias_variable('bias', [no_filters])
        V_reshape = tf.reshape(V, (-1, no_features))
        s = tf.slice(tf.shape(V), [0], [len(V.get_shape()) - 1])
        s = tf.concat([s, tf.stack([no_filters])], 0)
        result = tf.reshape(tf.matmul(V_reshape, W) + b, s)

        return result


def make_hierarchical_pooling54(V, A, factors, name=None):
    with tf.variable_scope(name, default_name='HierarchicalPool54') as scope:
        if V is None:
            return V, A
        reshape_V = tf.reshape(V, (-1, FLAGS.node_number, V.get_shape()[-1].value))
        reshape_V = batch_matmul(reshape_V, factors)
        result = tf.reshape(reshape_V, (-1, 54, V.get_shape()[-1].value))
        # self.fMRI_V = tf.Print(self.fMRI_V, [tf.shape(self.fMRI_V)])
        # self.fMRI_A = tf.Print(self.fMRI_A, [tf.shape(self.fMRI_A)])

        result_A = tf.reshape(A, (-1, FLAGS.node_number))
        # result_A = tf.Print(result_A,[tf.shape(result_A)])
        result_A = tf.matmul(result_A, factors)
        # result_A = tf.Print(result_A, [tf.shape(result_A)])
        result_A = tf.reshape(result_A, (tf.shape(A)[0], FLAGS.node_number, -1))
        # result_A = tf.Print(result_A, [tf.shape(result_A)])
        result_A = batch_matmul(result_A, factors)
        # result_A = tf.Print(result_A, [tf.shape(result_A)])
        result_A = tf.reshape(result_A, (tf.shape(A)[0], 54, A.get_shape()[2].value, 54))
        return result, result_A


def make_hierarchical_pooling14(V, A, factors, name=None):
    with tf.variable_scope(name, default_name='HierarchicalPool14') as scope:
        if V is None:
            return V, A
        reshape_V = tf.reshape(V, (-1, 54, V.get_shape()[-1].value))
        reshape_V = batch_matmul(reshape_V, factors)
        result = tf.reshape(reshape_V, (-1, 14, V.get_shape()[-1].value))

        result_A = tf.reshape(A, (-1, 54))
        result_A = tf.matmul(result_A, factors)
        result_A = tf.reshape(result_A, (tf.shape(A)[0], 54, -1))
        result_A = batch_matmul(result_A, factors)
        result_A = tf.reshape(result_A, (tf.shape(A)[0], 14, A.get_shape()[2].value, 14))
        return result, result_A


def make_st_pooling54(V, A, factors, name=None):
    with tf.variable_scope(name, default_name='HierarchicalPool54') as scope:
        if V is None:
            return V, A
        V_shape = V.get_shape()
        reshape_V = tf.reshape(V, (-1, V_shape[2], V_shape[-1]))
        reshape_V = batch_matmul(reshape_V, factors)

        result = tf.reshape(reshape_V, (-1, V_shape[1], 54, V_shape[-1]))

        A_shape = A.get_shape()
        result_A = tf.reshape(A, (-1, A_shape[-1]))
        result_A = tf.matmul(result_A, factors)
        result_A = tf.reshape(result_A, (-1, A_shape[2], A_shape[3] * 54))
        result_A = batch_matmul(result_A, factors)
        result_A = tf.reshape(result_A, (-1, A_shape[1], 54, A_shape[3], 54))
        return result, result_A


def make_st_pooling14(V, A, factors, name=None):
    with tf.variable_scope(name, default_name='HierarchicalPool14') as scope:
        if V is None:
            return V, A
        V_shape = V.get_shape()
        reshape_V = tf.reshape(V, (-1, V_shape[2], V_shape[3]))
        reshape_V = batch_matmul(reshape_V, factors)
        result = tf.reshape(reshape_V, (-1, V_shape[1], 14, V_shape[3]))

        A_shape = A.get_shape()
        result_A = tf.reshape(A, (-1, A_shape[-1]))
        result_A = tf.matmul(result_A, factors)
        result_A = tf.reshape(result_A, (-1, A_shape[2], A_shape[3] * 14))
        result_A = batch_matmul(result_A, factors)
        result_A = tf.reshape(result_A, (-1, A_shape[1], 14, A_shape[3], 14))
        return result, result_A

def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False,name=None):
    with tf.variable_scope(name, default_name='my_att') as scope:
        conv1d = tf.layers.conv1d
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        bias =tf.get_variable('gat_bias',vals.shape[2], initializer=tf.constant_initializer(0.1), dtype = tf.float32)
        ret = vals+bias

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation

def attn_head_shared(seq_1, seq_2, out_sz, bias_mat_1, bias_mat_2, activation, seed, in_drop=0.0, coef_drop=0.0, residual=False,name=None):
    with tf.variable_scope(name, default_name='my_att') as scope:
        conv1d = tf.layers.conv1d
        if in_drop != 0.0:
            seq_1 = tf.nn.dropout(seq_1, 1.0 - in_drop, seed=seed)
            seq_2 = tf.nn.dropout(seq_2, 1.0 - in_drop, seed=seed)

        conv1d_1 = tf.layers.conv1d
        seq_fts_1 = conv1d_1(seq_1, out_sz, 1, use_bias=False)
        seq_fts_2 = conv1d_1(seq_2, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        conv1d_2 = tf.layers.conv1d
        conv1d_3 = tf.layers.conv1d

        f_1_1 = conv1d_2(seq_fts_1, 1, 1)
        f_1_2 = conv1d_3(seq_fts_1, 1, 1)
        logits_1 = f_1_1 + tf.transpose(f_1_2, [0, 2, 1])
        coefs_1 = tf.nn.softmax(tf.nn.leaky_relu(logits_1) + bias_mat_1)

        f_2_1 = conv1d_2(seq_fts_2, 1, 1)
        f_2_2 = conv1d_3(seq_fts_2, 1, 1)
        logits_2 = f_2_1 + tf.transpose(f_2_2, [0, 2, 1])
        coefs_2 = tf.nn.softmax(tf.nn.leaky_relu(logits_2) + bias_mat_2)

        if coef_drop != 0.0:
            coefs_1 = tf.nn.dropout(coefs_1, 1.0 - coef_drop, seed=seed+1)
            coefs_2 = tf.nn.dropout(coefs_2, 1.0 - coef_drop, seed=seed+1)

        if in_drop != 0.0:
            seq_fts_1 = tf.nn.dropout(seq_fts_1, 1.0 - in_drop, seed=seed+2)
            seq_fts_2 = tf.nn.dropout(seq_fts_2, 1.0 - in_drop, seed=seed+2)

        vals_1 = tf.matmul(coefs_1, seq_fts_1)
        vals_2 = tf.matmul(coefs_2, seq_fts_2)
        bias =tf.get_variable('gat_bias',vals_1.shape[2], initializer=tf.constant_initializer(0.1), dtype = tf.float32)
        ret_1 = vals_1+bias
        ret_2 = vals_2+bias

        # residual connection
        if residual:
            if seq_1.shape[-1] != ret_1.shape[-1]:
                ret_1 = ret_1 + conv1d(seq_1, ret_1.shape[-1], 1) # activation
            if seq_2.shape[-1] != ret_2.shape[-1]:
                ret_2 = ret_2 + conv1d(seq_2, ret_2.shape[-1], 1) # activation
            else:
                ret_1 = ret_1 + seq_1
                ret_2 = ret_2 + seq_2

        return activation(ret_1), activation(ret_2)  # activation