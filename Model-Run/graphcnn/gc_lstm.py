from graphcnn.layers import *


def normalization(tens, scope=None):
    # https://github.com/philipperemy/tensorflow-multi-dimensional-lstm/blob/master/md_lstm.py
    assert (len(tens.get_shape()) == 2)
    m, v = tf.nn.moments(tens, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'norm', reuse=tf.AUTO_REUSE):
        scale = tf.get_variable('scale',
                                shape=[tens.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tens.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    ln_initial = (tens - m) / tf.sqrt(v + 1e-5)

    return ln_initial * scale + shift


class GCNLSTMCell(tf.contrib.rnn.LayerRNNCell):
    """ST-LSTM cell adapted from Basic LSTM cell.
    """

    def __init__(self, num_units, cell_num, initializer=None, input_shape_A=None, input_shape_V=None, attention_V=None,
                 do_norm=False, activation=None, reuse=None, name=None, dtype=None, **kwargs):
        """Initialize the parameters for an LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          initializer: (optional) The initializer to use for the weight and
            projection matrices.
          forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training. Must set it manually to `0.0` when restoring from
            CudnnLSTM trained checkpoints.
          activation: Activation function of the inner states.  Default: `tanh`. It
            could also be string that is within Keras activation function names.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
          name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
          dtype: Default dtype of the layer (default of `None` means use the type
            of the first input). Required when `build` is called before `call`.
          **kwargs: Dict, keyword named properties for common layer attributes, like
            `trainable` etc when constructing the cell from configs of get_config().
          When restoring from CudnnLSTM-trained checkpoints, use
          `CudnnCompatibleLSTMCell` instead.
        """
        super(GCNLSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        self.do_norm = do_norm
        self._num_units = num_units
        self._initializer = initializer

        self._activation = tf.tanh
        self.built = False
        self.W = "weight" + str(cell_num)
        self.W_I = "weight_I" + str(cell_num)
        self.b = "bias" + str(cell_num)
        self.attention_V = attention_V

        if input_shape_V is None:
            raise ValueError("Expected inputs_shape to be known")

    def __call__(self, inputs_V, inputs_A, inputs_preA, h_prev, c_prev, informativeness=None, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            
            if h_prev != None:
                # i = input_gate, fs = forget_gate_S, o = output_gate, c = new_state
                lstm_V = tf.concat([inputs_V, h_prev], 2)
                # inputs_A = 2 * inputs_A - inputs_preA
                # inputs_A = inputs_A + tf.cast(tf.not_equal(inputs_A, inputs_preA), tf.float32)
                # inputs_A = inputs_A + tf.cast(tf.equal(inputs_A, inputs_preA), tf.float32)
            else:
                lstm_V = inputs_V

            f = tf.sigmoid(self.graph_cnn("f", lstm_V, inputs_A))
            i = tf.sigmoid(self.graph_cnn("i", lstm_V, inputs_A))
            o = tf.sigmoid(self.graph_cnn("o", lstm_V, inputs_A))
            c = tf.sigmoid(self.graph_cnn("c", lstm_V, inputs_A))
            # f = tf.Print(f, [f[0, 0, :]], message="f is :")
            # i = tf.Print(i, [i[0, 0, :]], message="i is :")

            if self.do_norm:
                i = normalization(i, 'i/')
                f = normalization(f, 'f/')
                o = normalization(o, 'o/')
                c = normalization(c, 'c/')
            # New state
            if h_prev != None:
                c = i * c + f * c_prev
            else:
                c = i * c
            if self.do_norm:
                c = normalization(c, 'c/')
            # New hidden state
            no_node = inputs_V.get_shape()[1].value
            # attention_W = make_attention_variable_with_weight_decay("attention" + self.W, [1, self._num_units])
            # attention_W = make_attention_variable_with_weight_decay("attention" + self.W, [1])
            h = tf.nn.sigmoid(o) * self._activation(c)
            # h = tf.multiply(h, attention_W) + h
            return h, c

    def graph_cnn(self, name, inputs_V, inputs_A):
        no_A = inputs_A.get_shape()[2].value
        no_features = inputs_V.get_shape()[2].value
        W = make_variable_with_weight_decay(name + self.W, [no_features * no_A, self._num_units], stddev=math.sqrt(
            1.0 / (no_features * (no_A + 1) * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        W_I = make_variable_with_weight_decay(name + self.W_I, [no_features, self._num_units], stddev=math.sqrt(
            GraphCNNGlobal.GRAPHCNN_I_FACTOR / (no_features * (no_A + 1) * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        b = make_bias_variable(name + self.b, [self._num_units])

        # 加入节点attention
        # no_node = inputs_V.get_shape()[1].value
        # attention_V = make_attention_variable_with_weight_decay('attention_V', [1, no_node])
        # attention_V = tf.Print(attention_V, [attention_V], message='attentionV shape:', summarize=2880)
        inputs_V = tf.add(inputs_V, tf.multiply(inputs_V, self.attention_V))

        A_shape = inputs_A.get_shape()
        # tf.shape(inputes_A)
        A_reshape = tf.reshape(inputs_A, tf.stack([-1, A_shape[1] * no_A, A_shape[1]]))
        # tf.reshape(inputs_A, tf.stack([-1, A_shape[1] * no_A, A_shape[1]]))
        n = tf.matmul(A_reshape, inputs_V)
        n = tf.reshape(n, [-1, A_shape[1], no_A * no_features])
        result = batch_mat_mult(n, W) + batch_mat_mult(inputs_V, W_I) + b

        # result = tf.reshape(result,[V_shape[0],V_shape[1],V_shape[2],no_filters])
        return result


def gcnlstm_loop(lstm_size, input_data_V, input_data_A, no_filter, attention_V, if_concat=False, do_norm=False):
    """
    @param lstm_size: the hidden units
    @param input_data: the data to process of shape [batch,frames,joints,channels]
    returns the output of the lstm
    """

    with tf.variable_scope("GC-LSTM", reuse=tf.AUTO_REUSE):
        # Results list
        results = []
        cell = []
        # Create cells
        for i in range(lstm_size):
            cell.append(GCNLSTMCell(no_filter, cell_num=i, input_shape_A=input_data_A.get_shape(),
                                    input_shape_V=input_data_V.get_shape(), attention_V=attention_V,
                                    initializer=tf.truncated_normal_initializer, name="layer1", do_norm=do_norm))

        # Reorder inputs to (cell_num, batch_size, node_num, features)
        v = tf.transpose(input_data_V, [1, 0, 2, 3])
        a = tf.transpose(input_data_A, [1, 0, 2, 3, 4])

        # Controls the initial index
        h_prev = None
        c_prev = None

        for i in range(lstm_size):
            # Update context
            h_prev, c_prev = cell[i].__call__(v[i, :, :, :], a[i, :, :, :, :], a[i - 1, :, :, :, :], h_prev, c_prev)

            results.append(h_prev)

        results_shape = results[0].get_shape()
        results = tf.reshape(results, [lstm_size, -1, input_data_V.get_shape()[2].value, results_shape[2]])
        results = tf.transpose(results, [1, 0, 2, 3])

        # results = tf.Print(results, [results[0, 0, 0, :]], message="result1 is :")
        # results = tf.Print(results, [results[1, 0, 0, :]], message="result2 is :")
        if if_concat == True:
            results = tf.reduce_sum(results, axis=1)
        # Return the output
        return results
