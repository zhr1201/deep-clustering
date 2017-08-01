'''
Script to train the model
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf
import ipdb
from datagenerator2 import DataGenerator
from model import Model

from GlobalConstont import *


# the .pkl file lists of data set
pkl_list = ['../dcdata/' + str(i) + '.pkl' for i in range(1, 12)]
val_list = ['../dcdata/val.pkl']
sum_dir = 'sum'
train_dir = 'train'

lr = 1e-3
n_hidden = 300
max_steps = 20000000
batch_size = 128


def train():
    with tf.Graph().as_default():
        # dropout keep probability
        p_keep_ff = tf.placeholder(tf.float32, shape=None)
        p_keep_rc = tf.placeholder(tf.float32, shape=None)
        # generator for training set and validation set
        data_generator = DataGenerator(pkl_list, batch_size)
        val_generator = DataGenerator(val_list, batch_size)
        # placeholder for input log spectrum, VAD info.,
        # and speaker indicator function
        in_data = tf.placeholder(
            tf.float32, shape=[batch_size, FRAMES_PER_SAMPLE, NEFF])
        VAD_data = tf.placeholder(
            tf.float32, shape=[batch_size, FRAMES_PER_SAMPLE, NEFF])
        Y_data = tf.placeholder(
            tf.float32, shape=[batch_size, FRAMES_PER_SAMPLE, NEFF, 2])
        # init the model
        BiModel = Model(n_hidden, batch_size, p_keep_ff, p_keep_rc)
        # build the net structure
        embedding = BiModel.inference(in_data)
        Y_data_reshaped = tf.reshape(Y_data, [-1, NEFF, 2])
        VAD_data_reshaped = tf.reshape(VAD_data, [-1, NEFF])
        # compute the loss
        loss = BiModel.loss(embedding, Y_data_reshaped, VAD_data_reshaped)
        # get the train operation
        train_op = BiModel.train(loss, lr)
        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.merge_all_summaries()
        sess = tf.Session()

        # either train from scratch or a trained model
        # saver.restore(sess, 'train/model.ckpt-492000')
        # val_loss = np.fromfile('val_loss').tolist()
        # init_step = 56001
        init = tf.initialize_all_variables()
        sess.run(init)
        init_step = 0

        summary_writer = tf.train.SummaryWriter(
            sum_dir, sess.graph)
        # val_loss = []

        last_epoch = data_generator.epoch

        for step in range(init_step, init_step + max_steps):
            start_time = time.time()
            data_batch = data_generator.gen_batch()
            # concatenate the samples into batch data
            in_data_np = np.concatenate(
                [np.reshape(item['Sample'], [1, FRAMES_PER_SAMPLE, NEFF])
                 for item in data_batch])
            VAD_data_np = np.concatenate(
                [np.reshape(item['VAD'], [1, FRAMES_PER_SAMPLE, NEFF])
                 for item in data_batch])
            VAD_data_np = VAD_data_np.astype('int')
            Y_data_np = np.concatenate(
                [np.reshape(item['Target'], [1, FRAMES_PER_SAMPLE, NEFF, 2])
                 for item in data_batch])
            Y_data_np = Y_data_np.astype('int')
            # train the model
            loss_value, _, summary_str = sess.run(
                [loss, train_op, summary_op],
                feed_dict={in_data: in_data_np,
                           VAD_data: VAD_data_np,
                           Y_data: Y_data_np,
                           p_keep_ff: 1 - P_DROPOUT_FF,
                           p_keep_rc: 1 - P_DROPOUT_RC})
            summary_writer.add_summary(summary_str, step)
            duration = time.time() - start_time
            # if np.isnan(loss_value):
                # import ipdb; ipdb.set_trace()
            assert not np.isnan(loss_value)
            if step % 100 == 0:
                # show training progress every 100 steps
                num_examples_per_step = batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch, epoch %d)')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch,
                                     data_generator.epoch))
            if step % 4000 == 0:
                # save model every 4000 steps
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            if last_epoch != data_generator.epoch:
                # doing validation every training epoch
                print('Doing validation')
                val_epoch = val_generator.epoch
                count = 0
                loss_sum = 0
                # average the validation loss
                while(val_epoch == val_generator.epoch):
                    count += 1
                    data_batch = val_generator.gen_batch()
                    in_data_np = np.concatenate(
                        [np.reshape(item['Sample'],
                         [1, FRAMES_PER_SAMPLE, NEFF])
                         for item in data_batch])
                    VAD_data_np = np.concatenate(
                        [np.reshape(item['VAD'], [1, FRAMES_PER_SAMPLE, NEFF])
                         for item in data_batch])
                    VAD_data_np = VAD_data_np.astype('int')
                    Y_data_np = np.concatenate(
                        [np.reshape(item['Target'],
                         [1, FRAMES_PER_SAMPLE, NEFF, 2])
                         for item in data_batch])
                    Y_data_np = Y_data_np.astype('int')
                    loss_value, = sess.run(
                        [loss],
                        feed_dict={in_data: in_data_np,
                                   VAD_data: VAD_data_np,
                                   Y_data: Y_data_np,
                                   p_keep_ff: 1,
                                   p_keep_rc: 1})
                    loss_sum += loss_value
                val_loss.append(loss_sum / count)
                print ('validation loss: %.3f' % (loss_sum / count))
                np.array(val_loss).tofile('val_loss')

            last_epoch = data_generator.epoch


print('%s start' % datetime.now())
train()
