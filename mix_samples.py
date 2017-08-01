'''
Script to mix two testing samples
'''
import librosa
import numpy as np


# provide the wav name and mix
# speech1 = '/media/nca/data/raw_data/speech_train_r/FCMM0/TRAIN_DR2_FCMM0_SI1957.WAV'
# speech2 = '/media/nca/data/raw_data/speech_train_r/FKLC0/TRAIN_DR4_FKLC0_SX355.WAV'
speech1 = '/media/nca/data/raw_data/speech_test/TEST_DR2_FJRE0_SA1.WAV'
speech2 = '/media/nca/data/raw_data/speech_test/TEST_DR2_MGWT0_SI1539.WAV'

data1, _ = librosa.load(speech1, sr=8000)
data2, _ = librosa.load(speech2, sr=8000)
mix = data1[:len(data2)] + data2[:len(data1)]
librosa.output.write_wav('mix.wav', mix, 8000)
