import os
import librosa
import soundfile as sf
import numpy as np
import glob
import pandas as pd
import math

def get_features(file_name):

    if file_name: 
        X, sample_rate = sf.read(file_name, dtype='float32')

    # mfcc (mel-frequency cepstrum coefficient)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    return mfccs_scaled

def get_features_spectrogram(file_name):

    if file_name: 
        X, sample_rate = sf.read(file_name, dtype='float32')

    # mel-spectrogram
    spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=2048)
    spectrogram_scaled = np.mean(spectrogram.T, axis=0)
    return spectrogram_scaled

def get_db(file_name):
    if file_name:
        X, sample_rate = sf.read(file_name, dtype='float32')

    # extract decibels
    # calculate root mean squares value
    rms = np.sqrt(np.mean(X**2))

    # transform amplitude's rms to decibel
    decibel = 20*math.log(rms/0.00001, 10)
    #sgram = librosa.stft(X)
    #sgram_mag, _ = librosa.magphase(sgram)
    #mel_scale = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
    #mel_gram = librosa.amplitude_to_db(mel_scale, ref=np.max)
    return decibel

def extract_features():

    # path to dataset containing 10 subdirectories of .ogg files
    sub_dirs = os.listdir('dataset')
    sub_dirs.sort()
    features_list = []
    for label, sub_dir in enumerate(sub_dirs):  
        for file_name in glob.glob(os.path.join('dataset',sub_dir,"*.ogg")):
            print("Extracting file ", file_name)
            try:
                mfccs = get_features(file_name)
            except Exception as e:
                print("Extraction error")
                continue
            features_list.append([mfccs,label])

    features_df = pd.DataFrame(features_list,columns = ['feature','class_label'])
    print(features_df.head())    
    return features_df