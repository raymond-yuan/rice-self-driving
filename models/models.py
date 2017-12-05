import tensorflow as tf
import numpy as np
from data_utils import BatchGenerator
from config import *


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

        net = slim.fully_connected(tf.reshape(net, [batch_size, seq_len, -1]), 128, activation_fn=tf.nn.relu)
        # net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        # net = slim.fully_connected(net, 128, activation_fn=tf.nn.relu)
        # net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
        # net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        # net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu)
        # net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        # net = slim.fully_connected(net, 128, activation_fn=None)
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

class Model():
    def make_model(self, mean, std):
        pass

    def do_epoch(self, session, sequences, mode):
        pass

class cnn(Model):
    def __init__(self):
        pass

    def make_model(self, mean, std):
        

class Komada(Model):
    def __init__(self, graph, mean, std):
        self.global_train_step = 0
        self.global_valid_step = 0

        self.KEEP_PROB_TRAIN = 0.35

        self.train_writer = tf.summary.FileWriter('v3/train_summary', graph=graph)
        self.valid_writer = tf.summary.FileWriter('v3/valid_summary', graph=graph)

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

        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    def do_epoch(self, session, sequences, mode):
        test_predictions = {}
        valid_predictions = {}

        batch_generator = BatchGenerator(sequence=sequences, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
        print()
        print('batch enerate', batch_generator)
        print()
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
                self.valid_writer.add_summary(summary, self.global_valid_step)
                self.global_valid_step += 1
                feed_inputs = feed_inputs[:, LEFT_CONTEXT:].flatten()
                steering_targets = feed_targets[:, :, 0].flatten()
                model_predictions = model_predictions.flatten()
                stats = np.stack([steering_targets, model_predictions, (steering_targets - model_predictions) ** 2])
                for i, img in enumerate(feed_inputs):
                    valid_predictions[img] = stats[:, i]
            elif mode == "test":
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
