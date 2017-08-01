'''
File defining the global constants
'''
FRAMES_PER_SAMPLE = 100  # number of frames forming a chunk of data
SAMPLING_RATE = 8000
FRAME_SIZE = 256
NEFF = 129  # effective FFT points
# amplification factor of the waveform sig
AMP_FAC = 10000
MIN_AMP = 10000
# TF bins smaller than THRESHOLD will be
# considered inactive
THRESHOLD = 40
# embedding dimention
EMBBEDDING_D = 40
# prams for pre-whitening
GLOBAL_MEAN = 44
GLOBAL_STD = 15.5
# feed forward dropout prob
P_DROPOUT_FF = 0.5
# recurrent dropout prob
P_DROPOUT_RC = 0.2
