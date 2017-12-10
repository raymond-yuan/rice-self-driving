import tensorflow as tf
import numpy as np
from data_utils import BatchGenerator, ImageGenerator
from keras.models import Sequential
from config import *
from keras.layers import *


slim = tf.contrib.slim

layer_norm = lambda x: tf.contrib.layers.layer_norm(inputs=x, center=True, scale=True, activation_fn=None,
                                                    trainable=True)

def get_optimizer(loss, lrate):
    optimizer = tf.train.AdamOptimizer(learning_rate=lrate)
    gradvars = optimizer.compute_gradients(loss)
    gradients, v = list(zip(*gradvars))
    print([x.name for x in v])
    gradients, _ = tf.clip_by_global_norm(gradients, 15.0)
    return optimizer.apply_gradients(zip(gradients, v))


def apply_vision_simple(image, keep_prob, batch_size, seq_len, scope=None, reuse=None):
    video = tf.reshape(image, shape=[batch_size, LEFT_CONTEXT + seq_len, HEIGHT, WIDTH, CHANNELS])
    with tf.variable_scope(scope, 'Vision', [image], reuse=reuse):
        net = slim.convolution(video, num_outputs=64, kernel_size=[3, 12, 12], stride=[1, 6, 6], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        aux1 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128,
                                    activation_fn=None)

        net = slim.convolution(net, num_outputs=64, kernel_size=[2, 5, 5], stride=[1, 2, 2], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        aux2 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128,
                                    activation_fn=None)

        net = slim.convolution(net, num_outputs=64, kernel_size=[2, 5, 5], stride=[1, 1, 1], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        aux3 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128,
                                    activation_fn=None)

        net = slim.convolution(net, num_outputs=64, kernel_size=[2, 5, 5], stride=[1, 1, 1], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        # at this point the tensor 'net' is of shape batch_size x seq_len x ...
        aux4 = slim.fully_connected(tf.reshape(net, [batch_size, seq_len, -1]), 128, activation_fn=None)

        net = slim.fully_connected(tf.reshape(net, [batch_size, seq_len, -1]), 1024, activation_fn=tf.nn.relu)
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu)
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        net = slim.fully_connected(net, 128, activation_fn=None)
        return layer_norm(tf.nn.elu(net + aux1 + aux2 + aux3 + aux4))  # aux[1-4] are residual connections (shortcuts)


class SamplingRNNCell(tf.nn.rnn_cell.RNNCell):
    """Simple sampling RNN cell."""

    def __init__(self, num_outputs, use_ground_truth, internal_cell):
        """
        if use_ground_truth then don't sample
        """
        self._num_outputs = num_outputs
        self._use_ground_truth = use_ground_truth  # boolean
        self._internal_cell = internal_cell  # may be LSTM or GRU or anything

    @property
    def state_size(self):
        return self._num_outputs, self._internal_cell.state_size  # previous output and bottleneck state

    @property
    def output_size(self):
        return self._num_outputs  # steering angle, torque, vehicle speed

    def __call__(self, inputs, state, scope=None):
        (visual_feats, current_ground_truth) = inputs
        prev_output, prev_state_internal = state
        context = tf.concat([prev_output, visual_feats], 1)
        new_output_internal, new_state_internal = self._internal_cell(context,
                                                                prev_state_internal)  # here the internal cell (e.g. LSTM) is called
        new_output = tf.contrib.layers.fully_connected(
            inputs=tf.concat([new_output_internal, prev_output, visual_feats], 1),
            num_outputs=self._num_outputs,
            activation_fn=None,
            scope="OutputProjection")
        # if self._use_ground_truth == True, we pass the ground truth as the state; otherwise, we use the model's predictions
        return new_output, (current_ground_truth if self._use_ground_truth else new_output, new_state_internal)


class Model:
    def make_model(self, mean, std):
        raise NotImplementedError

    def do_epoch(self, session, sequences, labels, mode):
        raise NotImplementedError


class CNN(Model):
    def __init__(self, graph, mean, std):
        self.global_train_step = 0
        self.global_valid_step = 0
        self.KEEP_PROB_CONV_TRAIN = 0.75
        self.KEEP_PROB_FC_TRAIN = 0.5

        self.train_writer = tf.summary.FileWriter('deep-cnn/train_summary', graph=graph)
        self.valid_writer = tf.summary.FileWriter('deep-cnn/valid_summary', graph=graph)

        print('Building model')
        self.make_model(mean, std)
        print('finished making model')

    def make_model(self, mean, std):
        def weight_variable(shape):
            '''
            Initialize weights
            :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
            w: width of the filters
            h: height of the filters
            Cin: the number of the channels of the filters
            Cout: the number of filters
            :return: a tensor variable for weights with initial values
            '''
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

        def bias_variable(shape):
            '''
            Initialize biases
            :param shape: shape of biases, e.g. [Cout] where
            Cout: the number of filters
            :return: a tensor variable for biases with initial values
            '''

            # IMPLEMENT YOUR BIAS_VARIABLE HERE
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W):
            '''
            Perform 2-D convolution
            :param x: input tensor of size [N, W, H, Cin] where
            N: the number of images
            W: width of images
            H: height of images
            Cin: the number of channels of images
            :param W: weight tensor [w, h, Cin, Cout]
            w: width of the filters
            h: height of the filters
            Cin: the number of the channels of the filters = the number of channels of images
            Cout: the number of filters
            :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
            '''

            # IMPLEMENT YOUR CONV2D HERE
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            '''
            Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
            :param x: input data
            :return: the results of maxpooling (max-marginalized + downsampling)
            '''

            # IMPLEMENT YOUR MAX_POOL_2X2 HERE
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        def batch_norm(x, bias):
            gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]),
                                name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, list(range(len(x.shape) - 1)))
            # return tf.nn.batch_normalization(x, batch_mean, batch_var, bias, gamma, var_epsilon)
            return tf.nn.batch_normalization(x, batch_mean, batch_var, bias, gamma)

        # inputs
        self.inputs = tf.placeholder(shape=(None, HEIGHT, WIDTH, CHANNELS), dtype=tf.float32, name="inputs") # images
        self.preprocessed_inputs = tf.image.resize_images(self.inputs, (256, 140))

        print('Input Shape: {}\nDownsample: {}'.format(self.inputs.get_shape().as_list(),
                                                       self.preprocessed_inputs.get_shape().as_list())
              )
        self.targets = tf.placeholder(shape=(IMAGE_BATCH_SIZE,),
                                 dtype=tf.float32, name="targets")  # seq_len x batch_size x OUTPUT_DIM
        targets_normalized = (self.targets - mean[0]) / std[0]

        self.conv_dropout = tf.placeholder(tf.float32, name="conv_dropout")
        self.fc_dropout = tf.placeholder(tf.float32, name="fc_dropout")

        conv1 = weight_variable([3, 3, 3, 32])
        b1 = bias_variable([32])
        h1 = tf.nn.relu(conv2d(self.preprocessed_inputs, conv1) + b1)
        # d1 = tf.nn.dropout(h1, self.conv_dropout)
        # pool1 = max_pool_2x2(d1)

        conv2 = weight_variable([3, 3, 32, 32])
        b2 = bias_variable([32])
        h2 = tf.nn.relu(conv2d(h1, conv2) + b2)
        pool1 = max_pool_2x2(h2) # 25% dropout
        d1 = tf.nn.dropout(pool1, self.conv_dropout)

        conv3 = weight_variable([3, 3, 32, 64])
        b3 = bias_variable([64])
        h3 = tf.nn.relu(conv2d(d1, conv3) + b3)

        conv4 = weight_variable([3,3,64,64])
        b4 = bias_variable([64])
        h4 = tf.nn.relu(conv2d(h3, conv4) + b4)
        pool2 = max_pool_2x2(h4)
        d2 = tf.nn.dropout(pool2, self.conv_dropout)

        # None, 64, 35, 64
        print('POOL ', d2.get_shape().as_list())
        new_shape = 1
        for s in d2.get_shape().as_list()[1:]:
            new_shape *= s
        fc1 = weight_variable([new_shape, 512])
        bfc1 = bias_variable([512])
        pool2_flat = tf.reshape(d2, [-1, new_shape])
        h3 = tf.nn.relu(tf.matmul(pool2_flat, fc1) + bfc1)
        d3 = tf.nn.dropout(h3, self.fc_dropout)

        fc2 = weight_variable([512, 1])
        bfc2 = bias_variable([1])
        self.steering_predictions = tf.matmul(d3, fc2) + bfc2

        # Summary statistics
        # weights = [conv1, conv2, fc1, fc2]
        # bias = [b1,b2,b3,b4]
        # activs = [h1,h2,h3]
        # pools = [pool1,pool2]
        # with tf.name_scope('weights'):
        #     map(variable_summaries, weights)
        # with tf.name_scope('bias'):
        #     map(variable_summaries, bias)
        # with tf.name_scope('ReLU'):
        #     map(variable_summaries, activs)
        # with tf.name_scope('pools'):
        #     map(variable_summaries, pools)


        self.lr = 1e-3
        self.rmse = tf.sqrt(tf.reduce_sum(tf.squared_difference(targets_normalized, self.steering_predictions)))
        tf.summary.scalar('RMSE_Loss', self.rmse)
        self.optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(self.rmse)
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)


    def do_epoch(self, session, sequences, labels, mode):
        """
        batch generator will return np arrays
        """
        test_predictions = {}
        valid_predictions = {}
        batch_generator = ImageGenerator(sequence_X=sequences, sequence_Y=labels, batch_size=IMAGE_BATCH_SIZE)
        total_num_steps = batch_generator.get_total_steps()
        acc_loss = np.float128(0.0)
        for step in range(total_num_steps):
            feed_inputs, feed_targets, input_paths = next(batch_generator.next())
            # print('FINISHGEED MAKING BACH')
            feed_dict = {self.inputs: feed_inputs, self.targets: feed_targets}
            if mode == "train":
                feed_dict.update({self.conv_dropout: self.KEEP_PROB_CONV_TRAIN, self.fc_dropout: self.KEEP_PROB_FC_TRAIN})

                summary, _, loss = session.run([self.summary_op, self.optimizer, self.rmse], feed_dict=feed_dict)
                self.train_writer.add_summary(summary, self.global_train_step)
                self.global_train_step += 1
            elif mode == "valid":
                feed_dict.update({self.conv_dropout: 1.0, self.fc_dropout: 1.0})
                model_predictions, summary, loss = \
                    session.run([self.steering_predictions,
                                 self.summary_op,
                                 self.rmse
                                 ],
                                feed_dict=feed_dict)
                self.valid_writer.add_summary(summary, self.global_valid_step)
                self.global_valid_step += 1
                # feed_inputs = feed_inputs.flatten()
                steering_targets = feed_targets.flatten()
                model_predictions = model_predictions.flatten()
                stats = np.stack([steering_targets, model_predictions, (steering_targets - model_predictions) ** 2])
                for i, img_path in enumerate(input_paths):
                    valid_predictions[img_path] = stats[:, i]
            elif mode == "test":
                feed_dict.update({self.conv_dropout: 1.0, self.fc_dropout: 1.0})
                model_predictions = \
                    session.run([
                        self.steering_predictions
                    ],
                        feed_dict=feed_dict)
                model_predictions = model_predictions.flatten()
                for i, img_path in enumerate(input_paths):
                    test_predictions[img_path] = model_predictions[i]
            if mode != "test" and step % 100 == 0:
                acc_loss += loss
                print('\r', step + 1, "/", total_num_steps, np.sqrt(acc_loss / (step + 1)))
        print()
        return (np.sqrt(acc_loss / total_num_steps), valid_predictions) if mode != "test" else (None, test_predictions)


class Komada(Model):
    def __init__(self, graph, mean, std):
        self.global_train_step = 0
        self.global_valid_step = 0

        self.KEEP_PROB_TRAIN = 0.25

        self.train_writer = tf.summary.FileWriter('v4/train_summary', graph=graph)
        self.valid_writer = tf.summary.FileWriter('v4/valid_summary', graph=graph)

        self.make_model(mean, std)
        print('finished making model')
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            # print(shape)
            # print(len(shape))
            variable_parameters = 1
            for dim in shape:
                # print(dim)
                variable_parameters *= dim.value
            # print(variable_parameters)
            total_parameters += variable_parameters
        print(total_parameters)

    def make_model(self, mean, std):
        output_config = {}

        # inputs
        learning_rate = tf.placeholder_with_default(input=1e-4, shape=())
        self.keep_prob = tf.placeholder_with_default(input=1.0, shape=())
        aux_cost_weight = tf.placeholder_with_default(input=0.1, shape=())

        self.inputs = tf.placeholder(shape=(BATCH_SIZE, LEFT_CONTEXT + SEQ_LEN),
                                dtype=tf.string)  # pathes to png files from the central camera
        self.targets = tf.placeholder(shape=(BATCH_SIZE, SEQ_LEN, OUTPUT_DIM),
                                 dtype=tf.float32)  # seq_len x batch_size x OUTPUT_DIM

        targets_normalized = (self.targets - mean) / std

        input_images = tf.stack([tf.image.decode_png(tf.read_file(x))
                                 for x in tf.unstack(tf.reshape(self.inputs, shape=[(LEFT_CONTEXT + SEQ_LEN) * BATCH_SIZE]))])
        input_images = -1.0 + 2.0 * tf.cast(input_images, tf.float32) / 255.0
        input_images.set_shape([(LEFT_CONTEXT + SEQ_LEN) * BATCH_SIZE, HEIGHT, WIDTH, CHANNELS])
        visual_conditions_reshaped = apply_vision_simple(image=input_images, keep_prob=self.keep_prob,
                                                         batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
        visual_conditions = tf.reshape(visual_conditions_reshaped, [BATCH_SIZE, SEQ_LEN, -1])
        visual_conditions = tf.nn.dropout(x=visual_conditions, keep_prob=self.keep_prob)

        rnn_inputs_with_ground_truth = (visual_conditions, targets_normalized)
        rnn_inputs_autoregressive = (visual_conditions, tf.zeros(shape=(BATCH_SIZE, SEQ_LEN, OUTPUT_DIM), dtype=tf.float32))

        internal_cell = tf.nn.rnn_cell.LSTMCell(num_units=RNN_SIZE, num_proj=RNN_PROJ)
        cell_with_ground_truth = SamplingRNNCell(num_outputs=OUTPUT_DIM, use_ground_truth=True, internal_cell=internal_cell)
        cell_autoregressive = SamplingRNNCell(num_outputs=OUTPUT_DIM, use_ground_truth=False, internal_cell=internal_cell)

        def get_initial_state(complex_state_tuple_sizes):
            flat_sizes = tf.contrib.framework.nest.flatten(complex_state_tuple_sizes)
            init_state_flat = [tf.tile(
                multiples=[BATCH_SIZE, 1],
                input=tf.get_variable("controller_initial_state_%d" % i, initializer=tf.zeros_initializer, shape=([1, s]),
                                      dtype=tf.float32))
                for i, s in enumerate(flat_sizes)]
            init_state = tf.contrib.framework.nest.pack_sequence_as(complex_state_tuple_sizes, init_state_flat)
            return init_state

        def deep_copy_initial_state(complex_state_tuple):
            flat_state = tf.contrib.framework.nest.flatten(complex_state_tuple)
            flat_copy = [tf.identity(s) for s in flat_state]
            deep_copy = tf.contrib.framework.nest.pack_sequence_as(complex_state_tuple, flat_copy)
            return deep_copy

        controller_initial_state_variables = get_initial_state(cell_autoregressive.state_size)
        self.controller_initial_state_autoregressive = deep_copy_initial_state(controller_initial_state_variables)
        controller_initial_state_gt = deep_copy_initial_state(controller_initial_state_variables)

        with tf.variable_scope("predictor"):
            out_gt, self.controller_final_state_gt = tf.nn.dynamic_rnn(cell=cell_with_ground_truth,
                                                                  inputs=rnn_inputs_with_ground_truth,
                                                                  sequence_length=[SEQ_LEN] * BATCH_SIZE,
                                                                  initial_state=controller_initial_state_gt,
                                                                  dtype=tf.float32,
                                                                  swap_memory=True, time_major=False)
        with tf.variable_scope("predictor", reuse=True):
            out_autoregressive, self.controller_final_state_autoregressive = tf.nn.dynamic_rnn(cell=cell_autoregressive,
                                                                                          inputs=rnn_inputs_autoregressive,
                                                                                          sequence_length=[
                                                                                                              SEQ_LEN] * BATCH_SIZE,
                                                                                          initial_state=self.controller_initial_state_autoregressive,
                                                                                          dtype=tf.float32,
                                                                                          swap_memory=True,
                                                                                          time_major=False)

        mse_gt = tf.reduce_mean(tf.squared_difference(out_gt, targets_normalized))
        mse_autoregressive = tf.reduce_mean(tf.squared_difference(out_autoregressive, targets_normalized))
        self.mse_autoregressive_steering = tf.reduce_mean(
            tf.squared_difference(out_autoregressive[:, :, 0], targets_normalized[:, :, 0]))
        self.steering_predictions = (out_autoregressive[:, :, 0] * std[0]) + mean[0]

        total_loss = self.mse_autoregressive_steering + aux_cost_weight * (mse_gt + mse_autoregressive)

        self.optimizer = get_optimizer(total_loss, learning_rate)

        tf.summary.scalar("MAIN_TRAIN_METRIC_rmse_autoregressive_steering", tf.sqrt(self.mse_autoregressive_steering))
        tf.summary.scalar("rmse_gt", tf.sqrt(mse_gt))
        tf.summary.scalar("rmse_autoregressive", tf.sqrt(mse_autoregressive))

        output_config['train_step'] = self.optimizer
        output_config['preds'] = self.steering_predictions
        output_config['mse_autoreg_steering'] = self.mse_autoregressive_steering
        self.summaries = tf.summary.merge_all()
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    def do_epoch(self, session, sequences, labels, mode):
        test_predictions = {}
        valid_predictions = {}

        batch_generator = BatchGenerator(sequence=sequences, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
        total_num_steps = int(1 + (batch_generator.indices[1] - 1) / SEQ_LEN)
        controller_final_state_gt_cur, controller_final_state_autoregressive_cur = None, None
        acc_loss = np.float128(0.0)
        for step in range(total_num_steps):
            feed_inputs, feed_targets = batch_generator.next()
            feed_dict = {self.inputs: feed_inputs, self.targets: feed_targets}
            if controller_final_state_autoregressive_cur is not None:
                feed_dict.update({self.controller_initial_state_autoregressive:
                                      controller_final_state_autoregressive_cur})
            if controller_final_state_gt_cur is not None:
                feed_dict.update({self.controller_final_state_gt: controller_final_state_gt_cur})
            if mode == "train":
                feed_dict.update({self.keep_prob: self.KEEP_PROB_TRAIN})
                summary, _, loss, controller_final_state_gt_cur, controller_final_state_autoregressive_cur = \
                    session.run([self.summaries,
                                 self.optimizer,
                                 self.mse_autoregressive_steering,
                                 self.controller_final_state_gt,
                                 self.controller_final_state_autoregressive],
                                feed_dict=feed_dict)
                self.train_writer.add_summary(summary, self.global_train_step)
                self.global_train_step += 1
            elif mode == "valid":
                model_predictions, summary, loss, controller_final_state_autoregressive_cur = \
                    session.run([self.steering_predictions,
                                 self.summaries,
                                 self.mse_autoregressive_steering,
                                 self.controller_final_state_autoregressive
                                 ],
                                feed_dict=feed_dict)
                self.valid_writer.add_summary(summary, self.global_train_step)
                self.global_valid_step += 1
                feed_inputs = feed_inputs[:, LEFT_CONTEXT:].flatten()
                steering_targets = feed_targets[:, :, 0].flatten()
                model_predictions = model_predictions.flatten()
                stats = np.stack([steering_targets, model_predictions, (steering_targets - model_predictions) ** 2])
                for i, img in enumerate(feed_inputs):
                    valid_predictions[img] = stats[:, i]
            elif mode == "test":
                print("Performing test")
                model_predictions, controller_final_state_autoregressive_cur = \
                    session.run([
                        self.steering_predictions,
                        self.controller_final_state_autoregressive
                    ],
                        feed_dict=feed_dict)
                feed_inputs = feed_inputs[:, LEFT_CONTEXT:].flatten()
                model_predictions = model_predictions.flatten()
                for i, img in enumerate(feed_inputs):
                    test_predictions[img] = model_predictions[i]
            if mode != "test":
                acc_loss += loss
                print('\r', step + 1, "/", total_num_steps, np.sqrt(acc_loss / (step + 1)))
        print()
        return (np.sqrt(acc_loss / total_num_steps), valid_predictions) if mode != "test" else (None, test_predictions)
