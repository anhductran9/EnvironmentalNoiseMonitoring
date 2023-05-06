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

# Setup

In this repository, I trained Convolution Neural Network, Multi Layer Perceptron and SVM for sound classification.
I achieved classification accuracy of approx ~80%.
MFCC (mel-frequency cepstrum) feature is used to train models. 
Other features like short term fourier transform, chroma, melspectrogram can also be extracted.

The dataset is downloaded and is kept inside "dataset" folder. It has 10 different classes each containing 1000 .ogg files.
Visualize the dataset by running visualize_data.py. 
This script takes a .ogg file as input and converts it into .wav form. 
The waveform is visualized in the form of a plot. 

A sample wav file for each class has been generated and kept within sample folder for reference.

To train and classify, execute main.py as -

```
python main.py cnn  // for training CNN
python main.py mlp  // for training MLP
python main.py svm  // for training SVM
```

Internally main.py uses extract_features.py and nn.py (or svm.py) to create and train model.

Once training is done, the trained models are automatically saved in h5 format.