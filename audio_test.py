'''
Script to test a audio sample using trained model
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
import librosa
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import tensorflow as tf
import ipdb
import itertools
from numpy.lib import stride_tricks
import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt

from AudioSampleReader import AudioSampleReader
from model import Model

from GlobalConstont import *


# dir for test audio
data_dir = 'mix.wav'
# not useful during sample test
sum_dir = 'sum'
# dir to load model
train_dir = 'train'

lr = 0.00001  # not useful during test
n_hidden = 300  # hidden state size
batch_size = 1  # 1 for audio sample test
hop_size = 64
# oracle flag to decide if a frame need to be seperated
sep_flag = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 10
# oracle permutation to concatenate the chuncks of output frames
oracal_p = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 10
# import ipdb; ipdb.set_trace()


def stft(sig, frameSize, overlapFac=0.75, window=np.hanning):
    """ short time fourier transform of audio signal """
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    # samples = np.append(np.zeros(np.floor(frameSize / 2.0)), sig)
    samples = np.array(sig, dtype='float64')
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(
        samples,
        shape=(cols, frameSize),
        strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)


def out_put(N_frame):
    '''Use trained model to infer N _frame chuncks of
    frames of input audio'''
    with tf.Graph().as_default():
        # feed forward keep prob
        p_keep_ff = tf.placeholder(tf.float32, shape=None)
        # recurrent keep prob
        p_keep_rc = tf.placeholder(tf.float32, shape=None)
        # audio sample generator
        data_generator = AudioSampleReader(data_dir)
        # placeholder for model input
        in_data = tf.placeholder(
            tf.float32, shape=[batch_size, FRAMES_PER_SAMPLE, NEFF])
        # init the model
        BiModel = Model(n_hidden, batch_size, p_keep_ff, p_keep_rc)
        # make inference of embedding
        embedding = BiModel.inference(in_data)
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()
        # restore the model
        saver.restore(sess, 'train/model.ckpt-492000')
        tot_frame = N_frame * FRAMES_PER_SAMPLE
        # arrays to store output waveform
        out_audio1 = np.zeros([(tot_frame - 1) * hop_size + FRAME_SIZE])
        out_audio2 = np.zeros([(tot_frame - 1) * hop_size + FRAME_SIZE])
        mix = np.zeros([(tot_frame - 1) * hop_size + FRAME_SIZE])
        N_assign = 0

        # for every chunk of frames of data
        for step in range(N_frame):
            # import ipdb; ipdb.set_trace()
            data_batch = data_generator.gen_next()
            if data_batch is None:
                break
            # log spectrum info.
            in_data_np = np.concatenate(
                [np.reshape(item['Sample'], [1, FRAMES_PER_SAMPLE, NEFF])
                 for item in data_batch])
            # phase info.
            in_phase_np = np.concatenate(
                [np.reshape(item['Phase'], [1, FRAMES_PER_SAMPLE, NEFF])
                 for item in data_batch])
            # VAD info.
            VAD_data_np = np.concatenate(
                [np.reshape(item['VAD'], [1, FRAMES_PER_SAMPLE, NEFF])
                 for item in data_batch])

            # get inferred embedding using trained model
            # with keep prob = 1
            embedding_np, = sess.run(
                [embedding],
                feed_dict={in_data: in_data_np,
                           p_keep_ff: 1,
                           p_keep_rc: 1})
            # ipdb.set_trace()
            # get active TF-bin embedding according to VAD
            embedding_ac = [embedding_np[i, j, :]
                            for i, j in itertools.product(
                                range(FRAMES_PER_SAMPLE), range(NEFF))
                            if VAD_data_np[0, i, j] == 1]
            if(sep_flag[step] == 1):
                # if the frame need to be seperated
                # cluster the embeddings
                # import ipdb; ipdb.set_trace()
                if embedding_ac == []:
                    break
                kmean = KMeans(n_clusters=2, random_state=0).fit(embedding_ac)

            else:
                # if the frame don't need to be seperated
                # don't split the embeddings
                kmean = KMeans(n_clusters=1, random_state=0).fit(embedding_ac)
            mask = np.zeros([FRAMES_PER_SAMPLE, NEFF, 2])
            ind = 0
            if N_assign == 0:
                # if their is no existing speaker in previous frame
                center = kmean.cluster_centers_
                N_assign = center.shape[0]
            elif N_assign == 1:
                # if their is one speaker in previous frame
                center_new = kmean.cluster_centers_
                # assign the embedding for a speaker to the speaker with the
                # closest centroid in previous frames
                if center_new.shape[0] == 1:
                    # update and smooth the centroid for 1 speaker
                    center = 0.7 * center + 0.3 * center_new
                else:
                    # update and smooth the centroid for 2 speakers
                    N_assign = 2
                    # compute their relative affinity
                    cor = np.matmul(center_new, np.transpose(center))
                    # ipdb.set_trace()
                    if(cor[1] > cor[0]):
                        # rearrange their sequence if not consistant with
                        # previous frames
                        kmean.cluster_centers_ = np.array(
                            [kmean.cluster_centers_[1],
                             kmean.cluster_centers_[0]])
                        kmean.labels_ = (kmean.labels_ == 0).astype('int')
                    center = kmean.cluster_centers_
            else:
                # two speakers have appeared
                center_new = kmean.cluster_centers_
                cor = np.matmul(center_new[0, :], np.transpose(center))
                # rearrange their sequence if not consistant with previous
                # frames
                if(cor[1] > cor[0]):
                    if(sep_flag[step] == 1):
                        kmean.cluster_centers_ = np.array(
                            [kmean.cluster_centers_[1],
                             kmean.cluster_centers_[0]])
                        kmean.labels_ = (kmean.labels_ == 0).astype('int')
                    else:
                        kmean.labels_ = (kmean.labels_ == 1).astype('int')
                # need permutation of their order(Oracle)
                if(oracal_p[step]):
                    kmean.cluster_centers_ = np.array(
                        [kmean.cluster_centers_[1],
                         kmean.cluster_centers_[0]])
                    kmean.labels_ = (kmean.labels_ == 0).astype('int')
                else:
                    kmean.labels_ = (~kmean.labels_).astype('int')
                center = center * 0.7 + 0.3 * kmean.cluster_centers_

            # transform the clustering result and VAD info. into masks
            for i in range(FRAMES_PER_SAMPLE):
                for j in range(NEFF):
                    if VAD_data_np[0, i, j] == 1:
                        mask[i, j, kmean.labels_[ind]] = 1
                        ind += 1
            for i in range(FRAMES_PER_SAMPLE):
                # apply the mask and reconstruct the waveform
                tot_ind = step * FRAMES_PER_SAMPLE + i
                # ipdb.set_trace()
                # amp = (in_data_np[0, i, :] *
                #        data_batch[0]['Std']) + data_batch[0]['Mean']
                amp = in_data_np[0, i, :] * GLOBAL_STD + GLOBAL_MEAN
                out_data1 = (mask[i, :, 0] * amp *
                             VAD_data_np[0, i, :])
                out_data2 = (mask[i, :, 1] * amp *
                             VAD_data_np[0, i, :])
                out_mix = amp
                out_data1_l = 10 ** (out_data1 / 20) / AMP_FAC
                out_data2_l = 10 ** (out_data2 / 20) / AMP_FAC
                out_mix_l = 10 ** (out_mix / 20) / AMP_FAC

                out_stft1 = out_data1_l * in_phase_np[0, i, :]
                out_stft2 = out_data2_l * in_phase_np[0, i, :]
                out_stft_mix = out_mix_l * in_phase_np[0, i, :]

                con_data1 = out_stft1[-2:0:-1].conjugate()
                con_data2 = out_stft2[-2:0:-1].conjugate()
                con_mix = out_stft_mix[-2:0:-1].conjugate()

                out1 = np.concatenate((out_stft1, con_data1))
                out2 = np.concatenate((out_stft2, con_data2))
                out_mix = np.concatenate((out_stft_mix, con_mix))
                frame_out1 = np.fft.ifft(out1).astype(np.float64)
                frame_out2 = np.fft.ifft(out2).astype(np.float64)
                frame_mix = np.fft.ifft(out_mix).astype(np.float64)

                out_audio1[tot_ind * hop_size:tot_ind * hop_size + FRAME_SIZE] += frame_out1 * 0.5016
                out_audio2[tot_ind * hop_size:tot_ind * hop_size + FRAME_SIZE] += frame_out2 * 0.5016
                mix[tot_ind * hop_size:tot_ind * hop_size + FRAME_SIZE] += frame_mix * 0.5016

        librosa.output.write_wav('out_31.wav', out_audio1, SAMPLING_RATE)
        librosa.output.write_wav('out_32.wav', out_audio2, SAMPLING_RATE)


out_put(4)
