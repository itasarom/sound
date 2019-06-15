import numpy as np
import torch
import librosa
import librosa.display
import pandas as pd
from IPython.display import Audio
import seaborn as sns
import os
import librosa
import librosa.display

from scipy.signal import convolve, fftconvolve

from collections import defaultdict

import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.model_selection import train_test_split


min_db = -100
PREEMPHASIS = 0.97
ref_db = 20

# алгоритм Гриффин-Лим

def griffin_lim(S, num_iter=20):
    np.random.seed(42)
    # реализуйте алгоритм для добавления к вещественной матрице S фазы
    # на выходе нужно вернуть комплексную матрицу
    # ...    
    
    
    phi = 2 * np.pi * np.random.rand()
    H = istft(S * np.exp(1j * phi))
    
    for iteration in range(num_iter):
        phi = np.angle(stft(H))
        H = istft(S * np.exp(1j * phi))
    
    return stft(H)
#     return H

def load(config, path):
    signal, _ = librosa.load(path, sr=config.sampling_rate)
    return signal

def show(audio):
    return Audio(audio, rate=sample_rate)

def mel_to_linear(M):
    return np.dot(inv_mel_basis, M)

def inv_normalize(S):
    return S * -min_db + min_db

def db_to_amp(x):
    return 10**(x/20)


def de_emphasis(x):
    # Это эквивалентно scipy.signal.lfilter([1], [1, PREEMPHASIS], x)
    return scipy.signal.lfilter([1, PREEMPHASIS], [1, 0, -PREEMPHASIS**2], x)# ...
    

def inv_melspectrogram(M):
    M = inv_normalize(M)
    M = mel_to_linear(db_to_amp(M + ref_db))
    M = istft(griffin_lim(M))
    x = de_emphasis(M)
    return x





def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    # trim silence
    if 0 < len(y): # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y) # trim, top_db=default(60)

    # return y
    # make it unified length to conf.samples
    if len(y) > conf.samples: # long enough
        if trim_long_data:
#             y = y[0:0+123121523]
            # print(conf.samples, len(y))
            y = y[0:0+conf.samples]
    else: # pad blank
        padding = conf.samples - len(y)    # add padding at both ends
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')
    return y


def trim_and_mel(conf, audio):
    if len(audio) > conf.samples:
        audio = audio[:conf.samples]

    result = audio_to_melspectrogram(conf, audio)
    return result

def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(audio, 
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def show_melspectrogram(conf, mels, title='Log-frequency power spectrogram'):
    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 
                             sr=conf.sampling_rate, hop_length=conf.hop_length,
                            fmin=conf.fmin, fmax=conf.fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
    mels = audio_to_melspectrogram(conf, x)
    if debug_display:
        IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))
        show_melspectrogram(conf, mels)
    return mels


def read_columns():
    columns = pd.read_csv("./data/sample_submission.csv").columns[1:]
    column_encoder = {c:id for id, c in enumerate(columns)}

    return columns, column_encoder


def read_dataset(path, config, transform):
    meta = pd.read_csv(path + ".csv")
    all_data = []
    all_labels = []
    
    max_len = 0
    
    for id, (f, label) in tqdm.tqdm(enumerate(zip(meta.fname, meta.labels)), total=len(meta)):
        all_labels.append(label)
        all_data.append(transform(read_audio(config, os.path.join(path, f), trim_long_data=config.trim_long_data)))
        max_len = max(all_data[-1].shape[-1], max_len)

    
    y = np.zeros((len(all_data), len(config.columns)))
    for id, label in enumerate(all_labels):
        labels = label.split(",")
        for l in labels:
            y[id, config.column_encoder[l]] = 1.0

    
    
    
        
    
    return all_data, y, max_len

def read_test(path, config, transform):
    all_data = []
    max_len = 0
    meta = list(os.listdir(path))
    meta.sort()
    
    
    
    for id, f in tqdm.tqdm(enumerate(meta), total=len(meta)):
        all_data.append(transform(read_audio(config, os.path.join(path, f), trim_long_data=config.trim_long_data)))
        max_len = max(all_data[-1].shape[-1], max_len)
        
    y = np.zeros((len(all_data), len(columns)))
    
    return meta, all_data, y, max_len


def read_filters(path, config, transform):
    all_data = []
    max_len = 0
    meta = list(os.listdir(path))
    meta.sort()
    
    
    
    for id, f in tqdm.tqdm(enumerate(meta), total=len(meta)):
        pathname = os.path.join(path, f)
        audio, sr = librosa.load(pathname, sr=config.sampling_rate)
        all_data.append(transform(audio))

    return meta, all_data


def apply_filter(sound, filter):
    modified_sound = fftconvolve(in2=filter, in1=sound) #/ np.linalg.norm(sound)
    return modified_sound


def predict(trainer, meta, loader):
    prediction = trainer.predict(loader)
    result = {col:prediction[:, id] for id, col in enumerate(columns)}
    result['fname'] = meta
    result = pd.DataFrame(result)
    result = result.set_index('fname').reset_index()

    return result