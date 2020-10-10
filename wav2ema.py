import librosa
import numpy as np
import os, sys, time
import python_speech_features
import logging, os

logging.disable(logging.WARNING)
logging.getLogger('tensorflow').disabled = True
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
import warnings
warnings.filterwarnings("ignore")

import keras
import tensorflow as tf
from keras.models import load_model
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

def output(wav_path='test.wav',model_path='Trained_AAI_Model.h5'):
    #mean and var of each articulator from a train set subject
    #mean = [16.08, 7.47, 17.11, -8.94, 5.62, -52.91, -13.05, 2.40, -22.42, 6.75, -28.93, 7.61]
    #var = [0.71, 0.75, 0.81, 2.22, 0.47, 1.58, 2.14, 1.77, 2.20, 1.63, 2.19, 2.01]

    signal, rate = librosa.load(wav_path, sr=16000)
    m_t = librosa.feature.mfcc(signal, 16000, n_mfcc=12, hop_length=int(0.010*rate), n_fft=int(0.020*rate)).T
    model=load_model(model_path)
    model.summary()
    m_t=m_t[np.newaxis,:,:]
    PredEMA=model.predict(m_t);
    PredEMA=np.squeeze(PredEMA)
    ema = PredEMA#*var + mean
    print("saving file to test.npy")
    with open('test.npy', 'wb') as f:
        np.save(f, ema)
    return ema
output()
