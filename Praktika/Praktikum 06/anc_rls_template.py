#%%
import numpy as np
import scipy.io.wavfile
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import time


#%%
# Load audio data
fs, d = scipy.io.wavfile.read("noisy_sound.wav")
_, x = scipy.io.wavfile.read("noise_outside.wav")
N_samples = len(d)

#%%
M = 4
sigma = 0.01
l = 0.99

P = np.ones((M,M), dtype=float)*sigma
w = np.zeros(M, dtype=float)

x = np.pad(x, pad_width=(M,0))
d = np.pad(d, pad_width=(M,0))
e = np.zeros_like(x)

for n in range(M, M+N_samples):
    u = x[n:n-M:-1]

    k = (1/l*P@u)/(1+1/l*u.T@P@u)

    e[n] = d[n] - np.dot(w,u)

    w += k*e[n]
    P = 1/l*(P - np.outer(k, u.T@P))

#%%
sd.play(e, fs)

#%%
plt.figure(2)
plt.subplot(211)
plt.plot(d)
plt.plot(x)
plt.grid(True)

plt.subplot(212)
plt.plot(e)
plt.grid(True)
plt.show()

#%%