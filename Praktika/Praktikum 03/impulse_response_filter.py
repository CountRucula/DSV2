import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import sounddevice as sd
import scipy.signal

fs = 48e3  # Abtastrate
M = 12  # Memory des LFSR
L = 2**M-1  # Länge MLS- bzw. PN-Sequenz. PN12 dauert 84ms
Nrep = 10  # Anzahl Wiederholungen
W = 24  # Wortbreite
Trec = 1  # Aufnahmezeit

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

    # TODO:
    s[m] = (s[m-6] + s[m-8] + s[m-11] + s[m-12]) % 2 # p(X) = 1+X6+X8+X11+X12

# Convert binary to bipolar: 0 => 1 and 1 => -1
s = 1 - 2 * s

# Generate Nrep PN-sequences
x = np.tile(s, Nrep)

# Signalaufzeichnung
sd.default.device = [input_devID, output_devID]
sd.default.channels = 2
recordSig = sd.playrec(x/np.max(np.abs(x)), samplerate=fs, blocking=True)

no_of_samples = len(recordSig[:,0])

# Visualization
plt.figure(1)
ax1 = plt.subplot(211)
ax1.plot(recordSig[:int(5*fs),0])
ax2 = plt.subplot(212, sharex=ax1)
ax2.plot(recordSig[:int(5*fs),1])

# Berechnung der KKF
akf1 = scipy.signal.fftconvolve(np.flip(s), recordSig[:,0])
akf1 = akf1/np.max(np.abs(akf1))
akf2 = scipy.signal.fftconvolve(np.flip(s), recordSig[:,1])
akf2 = akf2/np.max(np.abs(akf2))
t = np.arange(len(akf1))/fs

# c) TODO: Extraction der Impulsantwort (Fester um Kreuzkorrelationsspitze)
peaks,_ = scipy.signal.find_peaks(akf2, height=0.5)
ixMax = peaks[0]
h = akf1[ixMax-10:ixMax+1000]

# TODO: Frequenzgangextraktion
N = len(h)
H_fourier = np.fft.fft(h)/N # FFT und Normalisierung
frq = np.arange(N)*fs/N
H_fourier = H_fourier[:N // 2]
frq = frq[:N // 2]  # Einseitiger Frequenzbereich

plt.figure(2)
plt.subplot(311)
plt.plot(t, akf2, t, akf1)
plt.plot(t[ixMax], akf2[ixMax], 'x')
plt.grid(True)
plt.legend(['AKF_2','AKF_1', 'ixPeal'])
plt.xlabel('t / s')

plt.subplot(312)
plt.plot(h)
plt.xlabel('Sample')
plt.ylabel('Impulse response')
plt.grid(True)


# TODO
plt.subplot(313)
plt.plot(frq, 10*np.log10(np.abs(H_fourier)))
plt.xlabel('Frequenz (Hz)')
plt.ylabel('|H(f)| [dB]')
plt.grid(True)

plt.show()
