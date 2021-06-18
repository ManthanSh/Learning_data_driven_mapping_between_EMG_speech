#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import numpy as np
import os
import pyworld
import pyworld as pw
import scipy
import scipy.io
import copy
#import ffmpeg
#import audioread


# In[2]:


FEATURE_DIM = 25
SAMPLE_RATE = 16000
FRAMES = 512
FFTSIZE = 1024
SPEAKERS_NUM = 1  #in our experiment, we use four speakers

EPSILON = 1e-10
MODEL_NAME = 'LSTM_model'
sr=16000
ispad=False


# In[3]:


def cal_mcep(wav_ori, fs=SAMPLE_RATE, frame_period=10, dim=FEATURE_DIM, fft_size=FFTSIZE):
    '''cal mcep given wav singnal
        the frame_period used only for pad_wav_to_get_fixed_frames
    '''

    wav = wav_ori
    #Harvest F0 extraction algorithm.
    f0, timeaxis = pyworld.harvest(wav, fs, f0_ceil=500.0,frame_period=frame_period)

    #CheapTrick harmonic spectral envelope estimation algorithm.
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs, fft_size=fft_size)

    #D4C aperiodicity estimation algorithm.
    ap = pyworld.d4c(wav, f0, timeaxis, fs, fft_size=fft_size)
    #feature reduction nxdim
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)
    #log
    coded_sp = coded_sp.T  # dim x n

    res = {
        'f0': f0,  #n
        'ap': ap,  #n*fftsize//2+1
        'sp': sp,  #n*fftsize//2+1
        'coded_sp': coded_sp,  #dim * n
    }
    return res


# In[5]:
# DataDir='../../../SPIRE_EMA/DataBase/'
#Subs=os.listdir(DataDir)
#Subs=['Divya', 'Navaneetha', 'Pavan' , 'Rashmi', 'Rohini' ,'Shaique','Shankar', 'Siddharth']


#Sub=Subs[ss]#'Anand_S'
#print(Sub)
#Type='Neutral'
datasetDir= '/home2/data/Manthan/008/101/splicedwav/' #'/home2/data/Manthan/101/splicedwav/'
McepDir= '/home2/data/Manthan/008/101/MCEP/' #'/home2/data/Manthan/101/MCEP/'
# BeginEndDir='../../../SPIRE_EMA/StartStopMat/'+Sub+'/';
# StartStopFile=os.listdir(BeginEndDir)
# StartStopMAt=scipy.io.loadmat(BeginEndDir+StartStopFile[0])
# BeginEnd=StartStopMAt['BGEN']
#os.makedirs(McepDir, exist_ok=True)
#     StartStopMAt=scipy.io.loadmat(datasetDir+'../Monika_N_BE.mat')
#     BeginEnd=StartStopMAt['BGEN']
Tcoded_sps = []
Tf0=[]

with os.scandir(datasetDir) as it_f:
    print(it_f)
    data =[]
    for onefile in it_f:
        #print(onefile.path)
        data.append(onefile.path)
    for one_file in data:
        filename = one_file.split('/')[-1].split('.')[0]
        print(filename)
        audio_wav, _ = librosa.load(one_file, sr=sr, mono=False, dtype=np.float64)
        print(audio_wav.shape)
        audio_wav=audio_wav[0,:]
        audio_wav=audio_wav.copy(order='C')
        print(audio_wav.shape)
        # Sid=int(filename[-3:len(filename)])-1
        # mcep_dict=cal_mcep(audio_wav[int(BeginEnd[0,Sid]*sr):int(BeginEnd[1,Sid]*sr)], fs=sr, frame_period=10, dim=FEATURE_DIM)
        mcep_dict=cal_mcep(audio_wav, fs=sr, frame_period=10, dim=FEATURE_DIM)
        Tf0.append(mcep_dict['f0'])
        Tcoded_sps.append(mcep_dict['coded_sp'])
        #file_path_z = f'{filename}'
        #print(f'save file: {file_path_z}')
        #np.savez(file_path_z, mcep_dict)
        scipy.io.savemat(McepDir+filename+'_stats.mat',mcep_dict)



# In[6]:


print(len(Tcoded_sps))


# In[7]:


coded_sps_concatenated = np.concatenate(Tcoded_sps, axis=1)
coded_sps_mean = np.mean(coded_sps_concatenated, axis=1, keepdims=False)
coded_sps_std = np.std(coded_sps_concatenated, axis=1, keepdims=False)


# In[8]:


log_f0s_concatenated = np.ma.log(np.concatenate(Tf0))
log_f0s_mean = log_f0s_concatenated.mean()
log_f0s_std = log_f0s_concatenated.std()


# In[9]:


tempdict = {'log_f0s_mean': log_f0s_mean, 'log_f0s_std': log_f0s_std, 'coded_sps_mean': coded_sps_mean, 'coded_sps_std': coded_sps_std}
OutDir=McepDir
# file_path_z = f'{OutDir}/{Sub}_{Type}'
#filename = os.path.join(etc_path, f'{one_speaker}-stats.npz')
np.savez(OutDir+'-stats.npz', tempdict)
scipy.io.savemat(file_path_z+'_stats.mat',tempdict)


# In[ ]:


#data#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
#CTC#https://www.dlology.com/blog/how-to-train-a-keras-model-to-recognize-variable-length-text/


# In[10]:







# In[ ]:




