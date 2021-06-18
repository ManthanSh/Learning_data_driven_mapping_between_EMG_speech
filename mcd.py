import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt
import pysptk
import librosa
import soundfile as sf
synth=scipy.io.loadmat("/home2/data/Manthan/UKA_Corpus/MCEP/008_Trial_3_18_June_BLSTMBatch32__LSTMunits_pred128__Sent_Norm_out_.mat")['Test_Predicted']
target=scipy.io.loadmat("/home2/data/Manthan/UKA_Corpus/MCEP/008_Trial_3_18_June_BLSTMBatch32__LSTMunits_pred128__Sent_Norm_out_.mat")['Test_target']

#mean=scipy.io.loadmat('/home2/data/Manthan/UKA_Corpus/output/mean_std_out_101_Kaldi.mat')['mean']
#std=scipy.io.loadmat('/home2/data/Manthan/UKA_Corpus/output/mean_std_out_101_Kaldi.mat')['std']
print(synth.shape)
#synth=np.squeeze(synth)
print(synth.shape)


#print(mean)
#print(std.shape)
#print(target[0][1][1])
#print((target[0][1][1]*std)+mean)
#print((synth[0][1][1]*std)+mean)
result=[]
CC=[]
for j in range(0,len(synth[0])):
  target1=[]
  synth1=[]
  for i in range(0,len(target[0][j])):
		#target[0][j][i]=((target[0][j][i])-mean)/std
		#synth[0][j][i]=(synth[0][j][i])*std+mean
    target1.append(target[0][j][i])
    synth1.append(synth[0][j][i])
    #print(np.corrcoef(target[0][j][i],synth[0][j][i]))
    CC.append(np.corrcoef(target[0][j][i],synth[0][j][i])[0][1])
  target1=np.array(target1)
  synth1=np.array(synth1)
  
  #synth1=(synth1-np.mean(synth1,axis=0))/np.std(synth1,axis=0)
  check=synth1-target1
  check=np.square(check)
	#print(check.shape)
  blah=np.sum(check,axis=1)
  blah=np.sqrt(blah)
	#print(len(blah))
	#print(len(blah))
  alpha=10*math.sqrt(2)/math.log(10)
  result.append(alpha*np.sum(blah)/len(blah))
	#print(alpha*np.sum(blah)/len(blah))

#scipy.io.savemat('Result_1.mat',{'mcd':result})
print(result)
"""
synth=synth[0]
target=target[0]


T=0
S=0

k=-1
for i in range(0,52):
	#k=k+1
	temp=len(synth[i])
	sub=synth[i]-target[i]
	A=np.square(sub[i])
	S=np.sum(A,axis=1)
	blah=np.sqrt(S)
	#print(blah.shape)
	for j in range(0,1818):
		k=k+1
		#print(k)
		mcd[k]=alpha*blah[j]

mcd=np.array(mcd)
scipy.io.savemat('/home2/data/Manthan/mcd.mat',{'mcd':mcd})


mcd_py=[]
for i in range(len(synth)):
    blah=pysptk.cdist(synth[i],target[i])
    mcd_py.append(blah)

print(mcd_py)
print('MeanMCD',np.mean(mcd_py))
print('Correlation Coefficient',np.mean(CC))

#print(synth[0].shape)
predicted=librosa.feature.inverse.mfcc_to_audio(target[0], n_mels=40, dct_type=2, norm='ortho', ref=1.0)
sf.write('/home2/data/Manthan/UKA_Corpus/output/pred.wav',predicted,16000)
print(predicted.shape)
plt.plot(synth[0][:][1])
plt.plot(target[0][:][1])
plt.legend(('pred','target'))
plt.show()
"""
alpha=0.340
import pyworld as pw
mcd=[]
n_mcep=25
for j in range(0,len(synth[0])):
    target1=[]
    synth1=[]
    synth[0][j] = synth[0][j].copy(order='C')
    synth[0][j]= synth[0][j].astype('double')
    target[0][j]=target[0][j].astype('double')
    X=target[0][j].copy(order='C')
    #print(X.shape)
    decoded_sp = pw.decode_spectral_envelope(synth[0][j],16000, fft_size=512*2)
    decoded_sp_org = pw.decode_spectral_envelope(X,16000, fft_size=512*2)
    Wmc = np.apply_along_axis(pysptk.sp2mc, 1, decoded_sp, n_mcep, alpha)
    mc = np.apply_along_axis(pysptk.sp2mc, 1, decoded_sp_org, n_mcep, alpha)
    mcd.append(pysptk.cdist(Wmc,mc))
    
print(mcd)
print(np.mean(mcd))