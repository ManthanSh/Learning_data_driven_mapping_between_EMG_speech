# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 11:07:36 2021

@author: Manthan Sharma
"""
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy.io
import scipy
from sklearn.metrics import mean_squared_error as mse
def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal))/old_freq
    sample_times = np.arange(0, times[-1], 1/new_freq)
    result = np.interp(sample_times, times, signal)
    return result

def apply_to_all(function, signal_array, *args, **kwargs):
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:,i], *args, **kwargs))
def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)
def double_average(x):
    assert len(x.shape) == 1
    f = np.ones(9)/9.0
    v = np.convolve(x, f, mode='same')
    w = np.convolve(v, f, mode='same')
    return w

def get_emg_features(emg_data, debug=False):
    xs = emg_data - emg_data.mean(axis=0, keepdims=True)
    #xs=xs/1170
    #x=emg_data
    #print(xs.shape)
    #xs = apply_to_all(subsample, x , 600, 10000)
    #print(xs.shape)
    frame_features = []
    for i in range(emg_data.shape[1]):
        x = xs[:,i]
        #x=remove_drift(x,600)
        #x=x[:,i]
        
        w = double_average(x)
        p = x - w
        r = np.abs(p)
        f_l=15
        h_l=6
        w_h = librosa.util.frame(w, frame_length=f_l, hop_length=h_l).mean(axis=0)
        p_w = librosa.feature.rms(w, frame_length=f_l, hop_length=h_l, center=False)
        p_w = np.squeeze(p_w, 0)
        p_r = librosa.feature.rms(r, frame_length=f_l, hop_length=h_l, center=False)
        p_r = np.squeeze(p_r, 0)
        z_p = librosa.feature.zero_crossing_rate(p, frame_length=f_l, hop_length=h_l, center=False)
        z_p = np.squeeze(z_p, 0)
        r_h = librosa.util.frame(r, frame_length=f_l, hop_length=h_l).mean(axis=0)
        #s = abs(librosa.stft(np.ascontiguousarray(x), n_fft=15, hop_length=6, center=False))

        #print(p_r)
        #print(r_h)
        #print(mse(p_r,r_h))
        
        if debug:
            plt.figure(figsize=(8,8))
            plt.subplot(6,1,1)
            plt.plot(x)
            plt.title('EMG'+' Channel: '+str(i+1))
            plt.subplot(6,1,2)
            plt.plot(w_h)
            plt.title('LF_Mean')
            plt.subplot(6,1,3)
            plt.plot(p_w)
            plt.title('LF_Power')
            plt.subplot(6,1,4)
            plt.plot(p_r)
            plt.title('HF_Power')
            plt.subplot(6,1,5)
            plt.plot(z_p)
            plt.title('HF_ZCR')
            plt.subplot(6,1,6)
            plt.plot(r_h)
            plt.title('HF_Mean')
            plt.tight_layout()

            #plt.subplot(7,1,7)
            #plt.imshow(s, origin='lower', aspect='auto', interpolation='nearest')

            plt.show()

        frame_features.append(np.stack([w_h, p_w, p_r, z_p, r_h], axis=1))
        #frame_features.append(s.T)

    frame_features = np.concatenate(frame_features, axis=1)
    return frame_features.astype(np.float32)

"""
file='C:/Users/Manthan Sharma/Desktop/IISc/M.tech Project/UKA_Trial_LabelsFixed/emg/002/101/splicedArray_e07_002_101_0001.mat'
data=scipy.io.loadmat(file)
data=data['ADC_modified']
data=data[:6,:]
data=np.transpose(data)
print(data.shape)

get_emg_features(data, debug=True)
"""