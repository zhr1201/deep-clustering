'''
Class DataGenerator
Script using DataGenerator to generate .pkl format data set
'''
import numpy as np
import librosa
import cPickle
from numpy.lib import stride_tricks
import ipdb
import os
import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt
from GlobalConstont import *


def stft(sig, frameSize, overlapFac=0.75, window=np.hanning):
    """ short time fourier transform of audio signal """
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    # samples = np.append(np.zeros(np.floor(frameSize / 2.0)), sig)
    samples = np.array(sig, dtype='float64')
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize))
    # zeros at end (thus samples can be fully covered by frames)
    # samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(
        samples,
        shape=(cols, frameSize),
        strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)


class DataGenerator(object):
    def __init__(self, data_dir, batch_size):
        '''preprocess the training data
        data_dir: dir containing the training data
                  format:root_dir + speaker_dir + wavfiles'''
        # get dirs for each speaker
        self.speakers_dir = [os.path.join(data_dir, i)
                             for i in os.listdir(data_dir)]
        self.n_speaker = len(self.speakers_dir)
        self.batch_size = batch_size
        self.speaker_file = {}
        self.epoch = 0

        # get the files in each speakers dir
        for i in range(self.n_speaker):
            wav_dir_i = [os.path.join(self.speakers_dir[i], file)
                         for file in os.listdir(self.speakers_dir[i])]
            for j in wav_dir_i:
                if i not in self.speaker_file:
                    self.speaker_file[i] = []
                self.speaker_file[i].append(j)
        # ipdb.set_trace()
        # self.reinit()

    def resample(self):
        '''Resample all the files, not always necessary'''
        for speaker in self.speaker_file:
            for file in self.speaker_file[speaker]:
                data, sr = librosa.load(file, SAMPLING_RATE)
                librosa.output.write_wav(file, data, SAMPLING_RATE)

    def reinit(self):
        '''Init the training data using the wav files'''
        self.speaker_file_match = {}
        # generate match dict
        for i in range(self.n_speaker):
            for j in self.speaker_file[i]:
                k = np.random.randint(self.n_speaker)
                # requiring different speaker
                while(i == k):
                    k = np.random.randint(self.n_speaker)
                # import ipdb; ipdb.set_trace()
                l = np.random.randint(len(self.speaker_file[k]))
                self.speaker_file_match[j] = self.speaker_file[k][l]

        self.samples = []
        self.ind = 0
        # for each file pair, generate their mixture and reference samples
        for i in self.speaker_file_match:
            j = self.speaker_file_match[i]
            speech_1, _ = librosa.core.load(i, sr=SAMPLING_RATE)
            # amp factor between -3 dB - 3 dB
            fac = np.random.rand(1)[0] * 6 - 3
            speech_1 = 10. ** (fac / 20) * speech_1
            speech_2, _ = librosa.core.load(j, sr=SAMPLING_RATE)
            fac = np.random.rand(1)[0] * 6 - 3
            speech_2 = 10. ** (fac / 20) * speech_2
            # mix
            length = min(len(speech_1), len(speech_2))
            speech_1 = speech_1[:length]
            speech_2 = speech_2[:length]
            speech_mix = speech_1 + speech_2
            # compute log spectrum for 1st speaker
            speech_1_spec = np.abs(stft(speech_1, FRAME_SIZE)[:, :NEFF])
            speech_1_spec = np.maximum(
                speech_1_spec, np.max(speech_1_spec) / MIN_AMP)
            speech_1_spec = 20. * np.log10(speech_1_spec * AMP_FAC)
            # same for the 2nd speaker
            speech_2_spec = np.abs(stft(speech_2, FRAME_SIZE)[:, :NEFF])
            speech_2_spec = np.maximum(
                speech_2_spec, np.max(speech_2_spec) / MIN_AMP)
            speech_2_spec = 20. * np.log10(speech_2_spec * AMP_FAC)
            # same for the mixture
            speech_mix_spec0 = stft(speech_mix, FRAME_SIZE)[:, :NEFF]
            speech_mix_spec = np.abs(speech_mix_spec0)
            # speech_phase = speech_mix_spec0 / speech_mix_spec
            speech_mix_spec = np.maximum(
                speech_mix_spec, np.max(speech_mix_spec) / MIN_AMP)
            speech_mix_spec = 20. * np.log10(speech_mix_spec * AMP_FAC)
            max_mag = np.max(speech_mix_spec)
            # if np.isnan(max_mag):
                # import ipdb; ipdb.set_trace()
            speech_VAD = (speech_mix_spec > (max_mag - THRESHOLD)).astype(int)
            # print 'mean:' + str(np.mean(speech_mix_spec)) + '\n'
            # print 'std:' + str(np.std(speech_mix_spec)) + '\n'
            speech_mix_spec = (speech_mix_spec - GLOBAL_MEAN) / GLOBAL_STD

            len_spec = speech_1_spec.shape[0]
            k = 0
            while(k + FRAMES_PER_SAMPLE < len_spec):
                sample_1 = speech_1_spec[k:k + FRAMES_PER_SAMPLE, :]
                sample_2 = speech_2_spec[k:k + FRAMES_PER_SAMPLE, :]
                # phase = speech_phase[k: k + FRAMES_PER_SAMPLE, :]
                sample_mix = speech_mix_spec[k:k + FRAMES_PER_SAMPLE, :]\
                    .astype('float16')
                # Y: indicator of the belongings of the TF bin
                # 1st speaker or second speaker
                Y = np.array(
                    [sample_1 > sample_2, sample_1 < sample_2]).astype('bool')
                Y = np.transpose(Y, [1, 2, 0])
                VAD = speech_VAD[k:k + FRAMES_PER_SAMPLE, :].astype('bool')
                sample_dict = {'Sample': sample_mix,
                               'VAD': VAD,
                               'Target': Y}
                self.samples.append(sample_dict)
                k = k + FRAMES_PER_SAMPLE
        # dump the generated sample list
        cPickle.dump(self.samples, open('val.pkl', 'wb'))
        self.tot_samp = len(self.samples)
        np.random.shuffle(self.samples)

    def gen_batch(self):
        '''Output a batch of training samples'''
        n_begin = self.ind
        n_end = self.ind + self.batch_size
        # ipdb.set_trace()
        if n_end >= self.tot_samp:
            self.ind = 0
            n_begin = self.ind
            n_end = self.ind + self.batch_size
            self.epoch += 1
            if self.epoch % 100 == 0:
                self.reinit()
        self.ind += self.batch_size
        return self.samples[n_begin:n_end]


if __name__ == '__main__':
    data_dir = '/media/nca/data/raw_data/speech_train_rs/'
    # data_dir = 'speech/'
    gen = DataGenerator(data_dir, 64)
    gen.reinit()
