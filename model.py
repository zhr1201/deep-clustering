'''
Class Model: model for the deep clustering speech seperation
'''
import numpy as np
import ipdb
import tensorflow as tf

from GlobalConstont import *

# from ln_lstm import LayerNormalizedLSTMCell
# from bnlstm import BNLSTMCell


class Model(object):
    def __init__(self, n_hidden, batch_size, p_keep_ff, p_keep_rc):
        '''n_hidden: number of hidden states
           p_keep_ff: forward keep probability
           p_keep_rc: recurrent keep probability'''
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        # if training:
        #     self.p_keep_ff = 1 - P_DROPOUT_FF
        #     self.p_keep_rc = 1 - P_DROPOUT_RC
        # else:
        #     self.p_keep_ff = 1
        #     self.p_keep_rc = 1
        self.p_keep_ff = p_keep_ff
        self.p_keep_rc = p_keep_rc
        # biases and weights for the last layer
        self.weights = {
            'out': tf.Variable(
                tf.random_normal([2 * n_hidden, EMBBEDDING_D * NEFF]))
        }
        self.biases = {
            'out': tf.Variable(
                tf.random_normal([EMBBEDDING_D * NEFF]))
        }

    def inference(self, x):
        '''The structure of the network'''
        # ipdb.set_trace()
        # four layer of LSTM cell blocks
        with tf.variable_scope('BLSTM1') as scope:
            # lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(
            #     self.n_hidden)
            # lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(
            #     self.n_hidden)
            lstm_fw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_fw_cell, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            lstm_bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_bw_cell, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell, lstm_bw_cell, x,
                sequence_length=[FRAMES_PER_SAMPLE] * self.batch_size,
                dtype=tf.float32)
            state_concate = tf.concat(2, outputs)
        with tf.variable_scope('BLSTM2') as scope:
            # lstm_fw_cell2 = tf.nn.rnn_cell.LSTMCell(
            #     self.n_hidden)
            # lstm_bw_cell2 = tf.nn.rnn_cell.LSTMCell(
            #     self.n_hidden)
            lstm_fw_cell2 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_fw_cell2 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_fw_cell2, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            lstm_bw_cell2 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_bw_cell2 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_bw_cell2, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            outputs2, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell2, lstm_bw_cell2, state_concate,
                sequence_length=[FRAMES_PER_SAMPLE] * self.batch_size,
                dtype=tf.float32)
            state_concate2 = tf.concat(2, outputs2)
        with tf.variable_scope('BLSTM3') as scope:
            lstm_fw_cell3 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_fw_cell3 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_fw_cell3, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            lstm_bw_cell3 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_bw_cell3 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_bw_cell3, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            outputs3, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell3, lstm_bw_cell3, state_concate2,
                sequence_length=[FRAMES_PER_SAMPLE] * self.batch_size,
                dtype=tf.float32)
            state_concate3 = tf.concat(2, outputs3)
        with tf.variable_scope('BLSTM4') as scope:
            lstm_fw_cell4 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_fw_cell4 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_fw_cell4, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            lstm_bw_cell4 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_bw_cell4 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_bw_cell4, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            outputs4, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell4, lstm_bw_cell4, state_concate3,
                sequence_length=[FRAMES_PER_SAMPLE] * self.batch_size,
                dtype=tf.float32)
            state_concate4 = tf.concat(2, outputs4)
        # one layer of embedding output with tanh activation function
        out_concate = tf.reshape(state_concate4, [-1, self.n_hidden * 2])
        emb_out = tf.matmul(out_concate,
                            self.weights['out']) + self.biases['out']
        emb_out = tf.nn.tanh(emb_out)
        reshaped_emb = tf.reshape(emb_out, [-1, NEFF, EMBBEDDING_D])
        # normalization before output
        normalized_emb = tf.nn.l2_normalize(reshaped_emb, 2)
        return normalized_emb

    def loss(self, embeddings, Y, VAD):
        '''Defining the loss function'''
        embeddings_rs = tf.reshape(embeddings, shape=[-1, EMBBEDDING_D])
        VAD_rs = tf.reshape(VAD, shape=[-1])
        # get the embeddings with active VAD
        embeddings_rsv = tf.transpose(
            tf.mul(tf.transpose(embeddings_rs), VAD_rs))
        embeddings_v = tf.reshape(
            embeddings_rsv, [-1, FRAMES_PER_SAMPLE * NEFF, EMBBEDDING_D])
        # get the Y(speaker indicator function) with active VAD
        Y_rs = tf.reshape(Y, shape=[-1, 2])
        Y_rsv = tf.transpose(
            tf.mul(tf.transpose(Y_rs), VAD_rs))
        Y_v = tf.reshape(Y_rsv, shape=[-1, FRAMES_PER_SAMPLE * NEFF, 2])
        # fast computation format of the embedding loss function
        loss_batch = tf.nn.l2_loss(
            tf.batch_matmul(tf.transpose(
                embeddings_v, [0, 2, 1]), embeddings_v)) - \
            2 * tf.nn.l2_loss(
                tf.batch_matmul(tf.transpose(
                    embeddings_v, [0, 2, 1]), Y_v)) + \
            tf.nn.l2_loss(
                tf.batch_matmul(tf.transpose(
                    Y_v, [0, 2, 1]), Y_v))
        loss_v = (loss_batch) / self.batch_size
        tf.scalar_summary('loss', loss_v)
        return loss_v

    def train(self, loss, lr):
        '''Optimizer'''
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8)
        # optimizer = tf.train.MomentumOptimizer(lr, 0.9)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 200)
        train_op = optimizer.apply_gradients(
            zip(gradients, v))
        return train_op
