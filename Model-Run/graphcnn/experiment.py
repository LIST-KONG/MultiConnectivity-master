from graphcnn.helper import *
from graphcnn.network import *
from graphcnn.network_description import *
from graphcnn.layers import *
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf
import glob
import time
from tensorflow.python.training import queue_runner
import scipy.io


# This function is used to create tf.cond compatible tf.train.batch alternative
def _make_batch_queue(input_data, capacity, num_threads=1):
    queue = tf.PaddingFIFOQueue(capacity=capacity, dtypes=[s.dtype for s in input_data],
                                shapes=[s.get_shape() for s in input_data])
    tf.summary.scalar("fraction_of_%d_full" % capacity,
                      tf.cast(queue.size(), tf.float32) *
                      (1. / capacity))
    enqueue_ops = [queue.enqueue(input_data)] * num_threads
    queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, enqueue_ops))
    return queue


# GraphCNNExperiment类负责：
# 1）设置和执行实验。
# 2）为实验提供helper functions(例如，获取accuracy)
class GraphCNNExperiment(object):
    # ### 初始化 函数
    # 功能：通过外部传入三个参数
    # 1）data_set_name
    # 2）model_name
    # 3）net_constructor：网络的构造
    def __init__(self, data_set_name, model_name, net_constructor):
        # Initialize all defaults
        self.data_set_name = data_set_name
        self.model_name = model_name
        self.num_iterations = 200
        self.iterations_per_test = 5
        self.display_iter = 5
        self.snapshot_iter = 1000000
        self.train_batch_size = 0
        self.test_batch_size = 0
        self.crop_if_possible = False  # True
        self.debug = False
        self.starter_learning_rate = 0.1
        self.learning_rate_exp = 0.1
        self.learning_rate_step = 1000
        # silent可控制是否进入print_ext函数主体。
        self.reports = {}
        self.silent = False
        # 优化器
        self.optimizer = 'momentum'

        self.net_constructor = net_constructor
        self.net = GraphCNNNetwork()
        self.net_desc = GraphCNNNetworkDescription()
        tf.reset_default_graph()

    # print_ext 函数
    # 功能：打印传入的数组
    # 注意：可用silent flag来控制是否执行函数主体。
    def print_ext(self, *args):
        if not self.silent:
            # 函数主体：打印传入的数组
            print_ext(*args)

    # get_max_accuracy 函数
    # 功能：获取训练的网络中（测试）准确率最高的网络。
    # 注意：SHOULD ONLY BE USED IF test_batch_size == ALL TEST SAMPLES，“控制变量”思想。
    def get_max_accuracy(self):
        tf.reset_default_graph()
        with tf.variable_scope('loss') as scope:
            max_acc_test = tf.Variable(tf.zeros([]), name="max_acc_test")
        saver = tf.train.Saver()
        with tf.Session() as sess:
            max_it = self.load_model(sess, saver)
            return sess.run(max_acc_test), max_it


    def save_loss_array(self, loss_array, data_set_name, iter_num):
        verify_dir_exists('./results/loss_array')
        res = ''
        for loss in loss_array:
            res = res + '{}\n'.format(loss)
        with open('./results/loss_array/%s_%d.txt' % (data_set_name,iter_num), 'w') as file:
            file.write(res)

    # run_kfold_experiments 函数
    # 功能：计算k-折确率
    # 输入：所有的训练和测试数据的真实标签、测试标签
    # 输出：准确率均值、均方差
    def run_kfold_experiments(self, no_folds=10, random_state=125, iter_num=0):
        acc = []
        sensitivity = []
        specificity = []
        attr = []
        
        total_max_acc = 0.0

        self.net_constructor.create_network(self.net_desc, [])
        desc = self.net_desc.get_description()
        self.print_ext('Running CV for:', desc)
        start_time = time.time()
        for i in range(no_folds):
            tf.reset_default_graph()
            self.set_kfold(no_folds=no_folds, fold_id=i,random_state=random_state)
            [cur_max, sensitivity_max, specificity_max, max_it, att], loss_array = self.run()
            if cur_max > total_max_acc:
                total_max_acc = cur_max
                self.save_loss_array(loss_array, self.data_set_name, iter_num)
            self.print_ext('Fold %d max accuracy: %g  sensitivity: %g specificity: %g at %d' % (
                i, cur_max, sensitivity_max, specificity_max, max_it))
            acc.append(cur_max)
            sensitivity.append(sensitivity_max)
            specificity.append(specificity_max)
            attr.append(att)
            path = './results/part/' + self.data_set_name + '-' + str(i) + '.mat'
            scipy.io.savemat(path, {'attr_set': att})

            verify_dir_exists('./results/')
            with open('./results/%s.txt' % self.data_set_name, 'a+') as file:
                file.write('%s\t%d-fold\t%.4f sensitivity\t%.4f specificity\t%.4f \n' % (
                    str(datetime.now()), i, cur_max, sensitivity_max, specificity_max))
            # attr_arr.append(attr)
        acc = np.array(acc)
        sensitivity = np.array(sensitivity)
        specificity = np.array(specificity)
        mean_acc = np.mean(acc) * 100
        std_acc = np.std(acc) * 100
        mean_sensitivity = np.mean(sensitivity) * 100
        std_sensitivity = np.std(sensitivity) * 100
        mean_specificity = np.mean(specificity) * 100
        std_specificity = np.std(specificity) * 100
        self.print_ext('Result is: %.2f (+- %.2f)' % (mean_acc, std_acc))
        self.print_ext('sensitivity is: %.2f (+- %.2f)' % (mean_sensitivity, std_sensitivity))
        self.print_ext('specificity is: %.2f (+- %.2f)' % (mean_specificity, std_specificity))

        verify_dir_exists('./results/')
        with open('./results/%s.txt' % self.data_set_name, 'a+') as file:
            file.write(
                '%s\t%s\t%d-fold\t%d seconds\t%.2f (+- %.2f)\tsensitivity is: %.2f (+- %.2f)\tspecificity is: %.2f (+- %.2f)\n' % (
                    str(datetime.now()), desc, no_folds, time.time() - start_time, mean_acc, std_acc, mean_sensitivity,
                    std_sensitivity, mean_specificity, std_specificity))
            return mean_acc, std_acc, mean_sensitivity, std_sensitivity, mean_specificity, std_specificity, attr

    # preprocess_data 函数
    def preprocess_data(self, data_set):
        self.fMRI_vertices = data_set[0].astype(np.float32)
        self.fMRI_adjacency = data_set[1].astype(np.float32)
        self.DTI_vertices = data_set[2].astype(np.float32)
        self.DTI_adjacency = data_set[3].astype(np.float32)
        self.dynamic_fMRI_vertices = data_set[4].astype(np.float32)
        self.dynamic_fMRI_adjacency = data_set[5].astype(np.float32)
        self.graph_labels = data_set[6].astype(np.int64)
        self.no_samples = self.graph_labels.shape[0]

    # set_kfold 函数
    # 功能：创建CV信息。
    # 输入：
    # 1）折数：no_folds
    # 2）折号：fold_id
    def set_kfold(self, no_folds=10, fold_id=0, random_state=125):
        inst = KFold(n_splits=no_folds, shuffle=True, random_state=random_state)
        self.fold_id = fold_id

        self.KFolds = list(inst.split(np.arange(self.no_samples)))
        self.train_idx, self.test_idx = self.KFolds[fold_id]
        self.no_samples_train = self.train_idx.shape[0]
        self.no_samples_test = self.test_idx.shape[0]
        self.print_ext('Data ready. no_samples_train:', self.no_samples_train, 'no_samples_test:', self.no_samples_test)

        if self.train_batch_size == 0:
            self.train_batch_size = self.no_samples_train
        if self.test_batch_size == 0:
            self.test_batch_size = self.no_samples_test
        self.train_batch_size = min(self.train_batch_size, self.no_samples_train)
        self.test_batch_size = min(self.test_batch_size, self.no_samples_test)

    # crop_single_sample 函数
    # 功能：分离各个样本，以提高性能
    # 时间：cropped before batch
    def crop_single_sample(self, single_sample):
        vertices = tf.slice(single_sample[0], np.array([0, 0, 0], dtype=np.int64),
                            tf.cast(tf.stack([single_sample[3][0], single_sample[3][1], -1, ]), tf.int64))
        vertices.set_shape([None, None, self.graph_vertices.shape[3]])
        adjacency = tf.slice(single_sample[1], np.array([0, 0, 0, 0], dtype=np.int64),
                             tf.cast(tf.stack([single_sample[3][0], single_sample[3][1], -1, single_sample[3][1]]),
                                     tf.int64))
        adjacency.set_shape([None, None, self.graph_adjacency.shape[3], None])
        mask = tf.ones([tf.shape(vertices)[0] * tf.shape(vertices)[1], 1])

        # V, A, labels, mask
        # mask：第axis=-1位置增加一个维度
        return [vertices, adjacency, single_sample[2], mask]

    # create_input_variable 函数
    def create_input_variable(self, input):
        for i in range(len(input)):
            # tf.placeholder：用于得到传递进来的真实的训练样本。
            placeholder = tf.placeholder(tf.as_dtype(input[i].dtype), shape=input[i].shape)
            # tf.Variable：主要在于一些可训练变量（trainable variables），比如模型的权重（weights，W）或者偏执值（bias）。
            # untrainable variables
            var = tf.Variable(placeholder, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            self.variable_initialization[placeholder] = input[i]
            input[i] = var
        return input

    # create_data 函数
    # 功能：创建train和test队列
    # 创建input_producers和batch queues
    def create_data(self):
        with tf.device("/cpu:0"):
            with tf.variable_scope('input') as scope:
                # 创建training队列 #
                with tf.variable_scope('train_data') as scope:
                    self.print_ext('Creating training Tensorflow Tensors')

                    # Create tensor with all training samples
                    training_samples = [self.fMRI_vertices, self.fMRI_adjacency, self.DTI_vertices, self.DTI_adjacency,
                                        self.dynamic_fMRI_vertices, self.dynamic_fMRI_adjacency, self.graph_labels]
                    training_samples = [s[self.train_idx, ...] for s in training_samples]

                    # if self.crop_if_possible == False:
                    # training_samples[3] = get_node_mask(training_samples[3], max_size=self.graph_vertices.shape[2])

                    # Create tf.constants
                    training_samples = self.create_input_variable(training_samples)

                    # Slice first dimension to obtain samples
                    single_sample = tf.train.slice_input_producer(training_samples, shuffle=True,
                                                                  capacity=self.train_batch_size)

                    # Cropping samples improves performance but is not required
                    if self.crop_if_possible:
                        self.print_ext('Cropping smaller graphs')
                        single_sample = self.crop_single_sample(single_sample)

                    # creates training batch queue
                    train_queue = _make_batch_queue(single_sample, capacity=self.train_batch_size * 2, num_threads=6)

                # 创建test队列 #
                with tf.variable_scope('test_data') as scope:
                    self.print_ext('Creating test Tensorflow Tensors')

                    # Create tensor with all test samples
                    test_samples = [self.fMRI_vertices, self.fMRI_adjacency, self.DTI_vertices, self.DTI_adjacency,
                                    self.dynamic_fMRI_vertices, self.dynamic_fMRI_adjacency, self.graph_labels]
                    test_samples = [s[self.test_idx, ...] for s in test_samples]

                    # If using mini-batch we will need a queue
                    if self.test_batch_size != self.no_samples_test:
                        # if self.crop_if_possible == False:
                        #     test_samples[3] = get_node_mask(test_samples[3], max_size=self.graph_vertices.shape[2])
                        test_samples = self.create_input_variable(test_samples)

                        single_sample = tf.train.slice_input_producer(test_samples, shuffle=True,
                                                                      capacity=self.test_batch_size)
                        if self.crop_if_possible:
                            single_sample = self.crop_single_sample(single_sample)

                        test_queue = _make_batch_queue(single_sample, capacity=self.test_batch_size * 2, num_threads=1)

                    # If using full-batch no need for queues
                    else:
                        # test_samples[3] = get_node_mask(test_samples[3], max_size=self.graph_vertices.shape[2])
                        test_samples = self.create_input_variable(test_samples)

                # obtain batch depending on is_training and if test is a queue

                if self.test_batch_size == self.no_samples_test:
                    return tf.cond(self.net.is_training, lambda: train_queue.dequeue_many(self.train_batch_size),
                                   lambda: test_samples)
                return tf.cond(self.net.is_training, lambda: train_queue.dequeue_many(self.train_batch_size),
                               lambda: test_queue.dequeue_many(self.test_batch_size))

    # create_loss_function 函数
    # Function called with the output of the Graph-CNN model
    # Should add the loss to the 'losses' collection and add any summaries needed (e.g. accuracy)
    def create_loss_function(self):
        with tf.variable_scope('loss') as scope:
            self.print_ext('Creating loss function and summaries')
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.net.result, labels=self.net.labels))
            correct_prediction = tf.cast(tf.equal(tf.argmax(self.net.result, 1), self.net.labels), tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)
            ones_like = tf.ones_like(self.net.labels)
            zeros_like = tf.zeros_like(self.net.labels)

            tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(self.net.labels, ones_like),
                                                      tf.equal(tf.argmax(self.net.result, 1), ones_like)),
                                       tf.float32))

            tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(self.net.labels, zeros_like),
                                                      tf.equal(tf.argmax(self.net.result, 1), zeros_like)),
                                       tf.float32))

            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(self.net.labels, zeros_like),
                                                      tf.equal(tf.argmax(self.net.result, 1), ones_like)),
                                       tf.float32))

            fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(self.net.labels, ones_like),
                                                      tf.equal(tf.argmax(self.net.result, 1), zeros_like)),
                                       tf.float32))

            sensitivity = tp / (tp + fn)
            specificity = tn / (fp + tn)

            # we have 2 variables that will keep track of the best accuracy obtained in training/testing batch
            # SHOULD ONLY BE USED IF test_batch_size == ALL TEST SAMPLES
            self.max_acc_train = tf.Variable(tf.zeros([]), name="max_acc_train")
            self.max_acc_test = tf.Variable(tf.zeros([]), name="max_acc_test")
            self.max_sensitivity_train = tf.Variable(tf.zeros([]), name="max_sensitivity_train")
            self.max_sensitivity_test = tf.Variable(tf.zeros([]), name="max_sensitivity_test")
            self.max_specificity_train = tf.Variable(tf.zeros([]), name="max_specificity_train")
            self.max_specificity_test = tf.Variable(tf.zeros([]), name="max_specificity_test")
            max_acc = tf.cond(self.net.is_training, lambda: self.max_acc_train, lambda: self.max_acc_test)
            max_sensitivity = tf.cond(self.net.is_training, lambda: tf.cond(tf.greater(accuracy, max_acc),
                                                                            lambda: tf.assign(
                                                                                self.max_sensitivity_train,
                                                                                sensitivity),
                                                                            lambda: self.max_sensitivity_train),
                                      lambda: tf.cond(tf.greater(accuracy, max_acc),
                                                      lambda: tf.assign(self.max_sensitivity_test, sensitivity),
                                                      lambda: self.max_sensitivity_test))
            max_specificity = tf.cond(self.net.is_training, lambda: tf.cond(tf.greater(accuracy, max_acc),
                                                                            lambda: tf.assign(
                                                                                self.max_specificity_train,
                                                                                specificity),
                                                                            lambda: self.max_specificity_train),
                                      lambda: tf.cond(tf.greater(accuracy, max_acc),
                                                      lambda: tf.assign(self.max_specificity_test, specificity),
                                                      lambda: self.max_specificity_test))
            max_acc = tf.cond(self.net.is_training,
                              lambda: tf.assign(self.max_acc_train, tf.maximum(self.max_acc_train, accuracy)),
                              lambda: tf.assign(self.max_acc_test, tf.maximum(self.max_acc_test, accuracy)))

            tf.add_to_collection('losses', cross_entropy)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('max_accuracy', max_acc)
            tf.summary.scalar('sensitivity', sensitivity)
            tf.summary.scalar('max_sensitivity', max_sensitivity)
            tf.summary.scalar('specificity', specificity)
            tf.summary.scalar('max_specificity', max_specificity)
            tf.summary.scalar('cross_entropy', cross_entropy)

            # if silent == false display these statistics:
            self.reports['accuracy'] = accuracy
            self.reports['max acc.'] = max_acc
            self.reports['max_sensitivity'] = max_sensitivity
            self.reports['max_specificity'] = max_specificity
            self.reports['cross_entropy'] = cross_entropy
            # self.reports['result'] = self.net.result
            self.reports['label'] = tf.argmax(self.net.result, 1)
            # self.reports['label_true'] = self.net.labels

    # check_model_iteration 函数
    # check if the model has a saved iteration and return the latest iteration step
    def check_model_iteration(self):
        latest = tf.train.latest_checkpoint(self.snapshot_path)
        if latest == None:
            return 0
        return int(latest[len(self.snapshot_path + 'model-'):])

    # load_model 函数
    # load_model if any checkpoint exist
    def load_model(self, sess, saver, ):
        latest = tf.train.latest_checkpoint(self.snapshot_path)
        if latest == None:
            return 0
        saver.restore(sess, latest)
        i = int(latest[len(self.snapshot_path + 'model-'):])
        self.print_ext("Model restored at %d." % i)
        return i

    # save_model 函数
    def save_model(self, sess, saver, i):
        latest = tf.train.latest_checkpoint(self.snapshot_path)
        if latest == None or i != int(latest[len(self.snapshot_path + 'model-'):]):
            self.print_ext('Saving model at %d' % i)
            verify_dir_exists(self.snapshot_path)
            result = saver.save(sess, self.snapshot_path + 'model', global_step=i)
            self.print_ext('Model saved to %s' % result)

    # Create graph (input, network, loss)
    # Handle checkpoints
    # Report summaries if silent == false
    # start/end threads
    def run(self):
        self.variable_initialization = {}
        loss_array = []

        self.print_ext('Training model "%s"!' % self.model_name)
        if hasattr(self, 'fold_id') and self.fold_id:
            self.snapshot_path = './snapshots/%s/%s/' % (self.data_set_name, self.model_name + '_fold%d' % self.fold_id)
            self.test_summary_path = './summary/%s/test/%s_fold%d' % (self.data_set_name, self.model_name, self.fold_id)
            self.train_summary_path = './summary/%s/train/%s_fold%d' % (
                self.data_set_name, self.model_name, self.fold_id)
        else:
            self.snapshot_path = './snapshots/%s/%s/' % (self.data_set_name, self.model_name)
            self.test_summary_path = './summary/%s/test/%s' % (self.data_set_name, self.model_name)
            self.train_summary_path = './summary/%s/train/%s' % (self.data_set_name, self.model_name)
        if self.debug:
            i = 0
        else:
            i = self.check_model_iteration()
        if i < self.num_iterations:
            self.print_ext('Creating training network')

            self.net.is_training = tf.placeholder(tf.bool, shape=())
            self.net.global_step = tf.Variable(0, name='global_step', trainable=False)

            input = self.create_data()
            self.net_constructor.create_network(self.net, input)
            self.create_loss_function()

            self.print_ext('Preparing training')
            loss = tf.add_n(tf.get_collection('losses')) * 100
            if len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
                loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            loss += self.net.loss * 0.0000001  # 0.00001xianzai

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.reports['loss'] = loss

            with tf.control_dependencies(update_ops):
                if self.optimizer == 'adam':
                    train_step = tf.train.AdamOptimizer().minimize(loss, global_step=self.net.global_step)
                else:
                    self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.net.global_step,
                                                                    self.learning_rate_step, self.learning_rate_exp,
                                                                    staircase=True)
                    train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(loss,
                                                                                              global_step=self.net.global_step)
                    self.reports['lr'] = self.learning_rate
                    tf.summary.scalar('learning_rate', self.learning_rate)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer(), self.variable_initialization)

                if self.debug == False:
                    saver = tf.train.Saver()
                    self.load_model(sess, saver)

                    self.print_ext('Starting summaries')
                    test_writer = tf.summary.FileWriter(self.test_summary_path, sess.graph)
                    train_writer = tf.summary.FileWriter(self.train_summary_path, sess.graph)

                summary_merged = tf.summary.merge_all()

                self.print_ext('Starting threads')
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                self.print_ext('Starting training. train_batch_size:', self.train_batch_size, 'test_batch_size:',
                               self.test_batch_size)
                wasKeyboardInterrupt = False
                try:
                    total_training = 0.0
                    total_testing = 0.0
                    start_at = time.time()
                    last_summary = time.time()
                    while i < self.num_iterations:
                        if i % self.snapshot_iter == 0 and self.debug == False:
                            self.save_model(sess, saver, i)
                        if i % self.iterations_per_test == 0:  # and i >= 700:
                            start_temp = time.time()
                            summary, reports = sess.run([summary_merged, self.reports],
                                                        feed_dict={self.net.is_training: 0})
                            total_testing += time.time() - start_temp
                            self.print_ext('Test Step %d Finished' % i)
                            for key, value in reports.items():
                                self.print_ext('Test Step %d "%s" = ' % (i, key), value)
                            if self.debug == False:
                                test_writer.add_summary(summary, i)
                        start_temp = time.time()
                        summary, _, reports = sess.run([summary_merged, train_step, self.reports],
                                                       feed_dict={self.net.is_training: 1})
                        
                        loss_array.append(reports['cross_entropy'])
                        
                        total_training += time.time() - start_temp
                        i += 1
                        if ((i - 1) % self.display_iter) == 0:
                            if self.debug == False:
                                train_writer.add_summary(summary, i - 1)
                            total = time.time() - start_at
                            self.print_ext(
                                'Training Step %d Finished Timing (Training: %g, Test: %g) after %g seconds' % (
                                    i - 1, total_training / total, total_testing / total, time.time() - last_summary))
                            for key, value in reports.items():
                                self.print_ext('Training Step %d "%s" = ' % (i - 1, key), value)
                            last_summary = time.time()
                        if (i - 1) % 100 == 0:
                            total_training = 0.0
                            total_testing = 0.0
                            start_at = time.time()
                    if i % self.iterations_per_test == 0:
                        summary = sess.run(summary_merged, feed_dict={self.net.is_training: 0})
                        if self.debug == False:
                            test_writer.add_summary(summary, i)
                        self.print_ext('Test Step %d Finished' % i)
                except Exception as err:
                    self.print_ext('Training interrupted at %d' % i)
                    self.print_ext(err)
                    wasKeyboardInterrupt = True
                    raisedEx = err
                finally:
                    if i > 0 and self.debug == False:
                        self.save_model(sess, saver, i)
                    self.print_ext('Training completed, starting cleanup!')
                    coord.request_stop()
                    coord.join(threads)
                    self.print_ext('Cleanup completed!')
                    if wasKeyboardInterrupt:
                        raise raisedEx

                return sess.run(
                    [self.max_acc_test, self.max_sensitivity_test, self.max_specificity_test, self.net.global_step,
                     self.net.attr], feed_dict={self.net.is_training: 0}), loss_array
        else:
            self.print_ext('Model "%s" already trained!' % self.model_name)
            return self.get_max_accuracy()
