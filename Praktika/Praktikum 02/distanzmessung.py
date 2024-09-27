import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import sounddevice as sd
import scipy.signal as sig
from scipy.signal import lfilter
 
fs = 48e3  # Abtastrate
M = 12  # Memory des LFSR
Nrep = 30  # Anzahl Wiederholungen
W = 24  # Wortbreite
Trec = 1  # Aufnahmezeit
 
L = (2**12) -1 # Länge MLS- bzw. PN-Sequenz.
 
# ID des USB-audio-interface suchen
#  (wichtig für audio-recording unten)
info = sd.query_devices()
input_devID, output_devID = None, None
for dev in info:
    if 'Focusrite' in dev['name'] or 'Scarlett' in dev['name']:
        if dev['max_input_channels'] > 0 and input_devID is None:
            input_devID = dev['index']
        if dev['max_output_channels'] > 0 and output_devID is None:
            output_devID = dev['index']
if input_devID is None or output_devID is None:
    print("USB audio interface not found. Please connect the appropriate device.")
    exit()
 
 
# Initialize the sequence
s = np.zeros(L)
s[:M] = 1  # Seed or initial value of the shift register
 
# Generate the PN sequence
for m in range(M, L):
    s[m] = (s[m-6] +s[m-8] + s[m-11] + s[m-12]) %2  # p(X) = 1+X6+X8+X11+X12
 
# Convert binary to bipolar: 0 => 1 and 1 => -1
s = 1 - 2 * s
 
# Generate Nrep PN-sequences
x = np.tile(s, Nrep)
 
# Signalaufzeichnung
sd.default.device = [input_devID, output_devID]
sd.default.channels = 2
 
recordSig = sd.playrec(x/np.max(np.abs(x)), samplerate=fs, blocking=True)   # Use sd.playrec(...)
no_of_samples = len(recordSig[:,0])
 
# Visualization
plt.figure(1)
plt.subplot(211)
plt.plot(recordSig[:int(5*fs),0])
plt.subplot(212)
plt.plot(recordSig[:int(5*fs),1])
 
#  Berechnung der AKF
akf1 = 0
akf2 = 0
 
akf1 = lfilter(s[::-1], 1,recordSig[:,0])
akf2 = lfilter(s[::-1], 1,recordSig[:,1])
 
t = np.arange(len(akf1))/fs
 
akf1 = akf1/np.max(np.abs(akf1))
akf2 = akf2/np.max(np.abs(akf2))
 
 
# Find largest peak in AKF1 and AKF2
t = np.arange(len(akf1))/fs
ixAkf2,_ = sig.find_peaks(akf2, 0.2)
ixAkf1,_ = sig.find_peaks(akf1, 0.2)
 
distance = (t[ixAkf1[0]] - t[ixAkf2[0]]) * 343  # [m]
 
plt.figure(2)
plt.plot(t, akf1, t, akf2)
plt.plot(t[ixAkf1[0]], akf1[ixAkf1[0]], '*')
plt.plot(t[ixAkf2[0]], akf2[ixAkf2[0]], 'o')
plt.grid(True)
plt.legend(['AKF_1','AKF_2'])
plt.xlabel('t / s')
plt.title("Distance = " + str(round(distance, 2)) + " m")
plt.show()
 
plt.figure(3)
plt.subplot(211)
plt.plot(t, akf1 )
plt.subplot(212)
plt.plot(t, akf2)
plt.grid(True)
plt.legend(['AKF_1','AKF_2'])
plt.xlabel('t / s')
plt.show()