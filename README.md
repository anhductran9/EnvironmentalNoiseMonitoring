# Environmental-Sound-Classification
Environmental-Sound-Classification 

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
Also use the customized environmental sound with 2100 environmental recordings (10 classes, 5 seconds per clip).

# Setup

In this repository, I trained Convolution Neural Network and Multi Layer Perceptron for sound classification.
I achieved classification accuracy of approx ~72%.
MFCC (mel-frequency cepstrum) feature is used to train models. 
Other features like short term fourier transform, chroma, melspectrogram can also be extracted.

To train and classify, execute main.py as -

```
python main.py cnn  // for training CNN
python main.py mlp  // for training MLP
```

Internally main.py uses extract_features.py and nn.py to create and train model.

Once training is done, the trained models are automatically saved in h5 format.
