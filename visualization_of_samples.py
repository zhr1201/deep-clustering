'''
Script for visualizing the embeddings
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
import matplotlib as mpl
mpl.use('agg')  # for saving figure on the server without UI
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import tensorflow as tf
import ipdb
import itertools

from AudioSampleReader import AudioSampleReader
from model import Model

from GlobalConstont import *


data_dir = 'mix.wav'
sum_dir = 'sum'
train_dir = 'train'

lr = 0.00001  # not needed for test
n_hidden = 300
batch_size = 1


def visualize(N_frame):
    with tf.Graph().as_default():
        # init the sample reader
        data_generator = AudioSampleReader(data_dir)
        # build the graph as the training script
        in_data = tf.placeholder(
            tf.float32, shape=[batch_size, FRAMES_PER_SAMPLE, NEFF])
        VAD_data = tf.placeholder(
            tf.float32, shape=[batch_size, FRAMES_PER_SAMPLE, NEFF])
        Y_data = tf.placeholder(
            tf.float32, shape=[batch_size, FRAMES_PER_SAMPLE, NEFF, 2])
        # init
        BiModel = Model(n_hidden, batch_size, False)
        # infer embedding
        embedding = BiModel.inference(in_data)
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()
        # restore a model
        saver.restore(sess, 'train/model.ckpt-68000')

        for step in range(N_frame):
            data_batch = data_generator.gen_next()
            if data_batch is None:
                break
            # concatenate the elements in sample dict to generate batch data
            in_data_np = np.concatenate(
                [np.reshape(item['Sample'], [1, FRAMES_PER_SAMPLE, NEFF])
                 for item in data_batch])
            VAD_data_np = np.concatenate(
                [np.reshape(item['VAD'], [1, FRAMES_PER_SAMPLE, NEFF])
                 for item in data_batch])
            embedding_np, = sess.run(
                [embedding],
                feed_dict={in_data: in_data_np,
                           VAD_data: VAD_data_np
                           })
            # only plot those embeddings whose VADs are active
            embedding_ac = [embedding_np[i, j, :]
                            for i, j in itertools.product(
                                range(FRAMES_PER_SAMPLE), range(NEFF))
                            if VAD_data_np[0, i, j] == 1]
            # ipdb.set_trace()

            kmean = KMeans(n_clusters=2, random_state=0).fit(embedding_ac)
            # visualization using 3 PCA
            pca_Data = PCA(n_components=3).fit_transform(embedding_ac)
            fig = plt.figure(1, figsize=(8, 6))
            ax = Axes3D(fig, elev=-150, azim=110)
            # ax.scatter(pca_Data[:, 0], pca_Data[:, 1], pca_Data[:, 2],
            #            c=kmean.labels_, cmap=plt.cm.Paired)
            ax.scatter(pca_Data[:, 0], pca_Data[:, 1], pca_Data[:, 2],
                       cmap=plt.cm.Paired)
            ax.set_title('Embedding visualization using the first 3 PCs')
            ax.set_xlabel('1st pc')
            ax.set_ylabel('2nd pc')
            ax.set_zlabel('3rd pc')
            plt.savefig('vis/' + str(step) + 'pca.jpg')


visualize(6)
