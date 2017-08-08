# A tensorflow implementation of deep clustering for speech seperation
This is a tensorflow implementation of the deep clustering paper: https://arxiv.org/abs/1508.04306
A few exmaples from the test set can be viewed in visualization_samples/ and speech_samples/

## Requirements
Python 2 and its packages:
  * tensorflow r0.11
  * numpy
  * scikit-learn
  * matplotlib
  * librosa
  
## File documentation
  * GlobalConstant.py: Gloabl constants.
  * datagenerator.py: Transform seperate speech files in a dir into .pkl format data set.
  * datagenerator2.py: A class to read the .pkl data set and generate batches of data for training the net.
  * model.py: A class defining the net structure.
  * train_net.py: Train the DC model.
  * mix_samples.py: Mix up two pieces of speech signals for test.
  * AudioSampleReader.py: Transform a .wav file into chunks of frames to be fed to the models during test.
  * visualization_of_samples.py: Visualize the active embedding points using PCA.
  * audio_test.py: Take in two speaker mix sample and seperate them.
  
## Training procedure
  1. Orgnize your speech data files as the following format:
      root_dir/speaker_id/speech_files.wav
  2. Make some changes dir of the datagenerator.py and run it, you may want to rename the .pkl file properly.
  3. Make dirs for write summaries and checkpoints, update your dirs in the train_net.py. The changes of the .pkl file list for      training and validation are also need to be made.
  4. Train the model.
  5. Generate some mixtures using mix_samples.py, and modify the checkpoints in audio_test.py.
  6. Enjoy yourself!
  
## Some other things
  The optimizer is not the same as that in the original paper, and also no 3 speaker mixture generator is provided, and we are moving on to the next stage of work and will not bother to do that. If you are interested and implemente that, we are glad to merge your branch.

## References
  https://arxiv.org/abs/1508.04306
  
