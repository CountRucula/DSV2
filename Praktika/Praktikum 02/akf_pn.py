# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# Parameters
fs = 48e3  # Sampling rate
M = 4  # Memory of the LFSR
Nrep = 4  # Number of periods

# TODO a)
L = 15  # Length or number of samples of the PN signal


# Generation of the periodic PN signal
s = np.zeros(L)
s[:M] = np.ones(M)  # seed or initial value shift register

for m in range(M, L):
    s[m] = (s[m-3] + s[m-4]) % 2  # TODO a): p(X) = 1+X3+X4


# Visualize PN sequence
plt.subplot(2, 1, 1)
plt.stem(np.arange(L), s)
plt.grid(True)
plt.xlabel('n')
plt.ylabel('s[n]')

s = 1 - 2*s  # binary2bipolar: 0 => 1 and 1 => -1

# Generate Nrep periods of the bipolar sequence s[n]
x = np.tile(s, Nrep)  # TODO: b)

# Calculation of the ACF
t = np.arange(Nrep*L) / fs

# TODO: b)
akf = lfilter(np.flip(s), 1, x)

plt.subplot(2, 1, 2)
plt.plot(t, akf, '-o')
plt.grid(True)
plt.xlabel('t / s')
plt.ylabel('AKF')

plt.show()
