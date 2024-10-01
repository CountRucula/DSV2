import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import sounddevice as sd
import scipy.signal
import scipy.io

# Parameters
fs = 48e3  # Abtastrate
M = 18  # Memory des LFSR
L = 2**M-1  # Länge MLS- bzw. PN-Sequenz. PN12 dauert 84ms
Nrep = 3  # Anzahl Wiederholungen
W = 24  # Wortbreite
Trec = 15  # Aufnahmezeit

# ID des USB-audio-interface suchen
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
    s[m] = (s[m-18] + s[m-11]) % 2

# Convert binary to bipolar: 0 => 1 and 1 => -1
s = 1 - 2 * s

# Generate Nrep PN-sequences
x = np.tile(s, Nrep)

# Signalaufzeichnung
print('Start recording.')
sd.default.device = [input_devID, output_devID]
sd.default.channels = 2
recordSig = sd.playrec(x/np.max(np.abs(x)), samplerate=fs, blocking=True)
no_of_samples = len(recordSig[:,0])

# Kontroll-Plot
plt.figure(1)
plt.subplot(211)
plt.plot(recordSig[:int(5*fs),0])
plt.subplot(212)
plt.plot(recordSig[:int(5*fs),1])

# Berechnung der AKF
akf1 = scipy.signal.fftconvolve(np.flip(s), recordSig[:,0])
akf1 = akf1/np.max(np.abs(akf1))
akf2 = scipy.signal.fftconvolve(np.flip(s), recordSig[:,1])
akf2 = akf2/np.max(np.abs(akf2))
t = np.arange(len(akf1))/fs

max_index1 = np.argmax(np.abs(akf1) > 0.99)  # max(abs(akf2)); finds maximum sometimes to late
h = akf1[max_index1 : max_index1+int(1.5*fs)-1]  # 1.5s duration of the impulseresponse
h = h - h[-1]  # subtract the DC
t_h = np.arange(len(h))/fs

plt.figure(2)
plt.subplot(311)
plt.plot(t_h, h)
plt.grid(True)
plt.xlabel('t / s')
plt.ylabel('h[n]')
plt.xlim([0, 1.5])
plt.title('Room-Impulse-Response (RIR): TS-Treppenhaus')

plt.subplot(312)
plt.plot(t_h, 20*np.log10(np.abs(h) + 1e-50))
plt.grid()
plt.xlabel('t / s')
plt.ylabel('h[n]/dB')
plt.axis([0, 1.5, -80, 0])
plt.title('RIR in dB')

plt.subplot(313)
NFFT = 1024
window = np.hamming(NFFT)
plt.specgram(h, window=window, noverlap=0, NFFT=NFFT, Fs=fs)
plt.colorbar(label="Power/frequency (dB/Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (kHz)")
plt.title('Spektrogramm der RIR TS-Treppenhaus')
plt.tight_layout()
plt.show()


# Save recorded signal to WAV file
#scipy.io.wavfile.write("impulse_response.wav", int(fs), h)
