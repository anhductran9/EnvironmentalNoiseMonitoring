# Environmental-Sound-Classification
Environmental-Sound-Classification for Noise Monitoring System

# Dependencies

1. Python
2. Keras
3. Librosa
4. sounddevice
5. SoundFile 
6. scikit-learn

# Dataset

Uses some of **ESC-10 dataset** for sound classification.
It is a labeled set of 400 environmental recordings (10 classes, 1000 clips, 5 seconds per clip). 
It is a subset of the larger **[ESC-50 dataset](https://github.com/karoldvl/ESC-50/)**.

# Setup

In this repository, I trained Convolutional Neural Network and Multi Layer Perceptron for sound classification.
MFCC (mel-frequency cepstrum) and Mel-Spectrogram feature is used to train models.
In the official model that has been chosen for the profect, which combining Convolutional Network and MFCC.
I can achieved classification accuracy of approximately ~71%.

The dataset is downloaded and is kept inside "dataset" folder. It has 10 different classes each containing 2100 .ogg files.

Internally main.py uses extract_features.py and nn.py to create and train model.

Once training is done, the trained models are automatically saved in h5 format.