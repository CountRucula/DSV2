import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa  #  https://stackoverflow.com/questions/52916266/cant-import-soundfile
import sounddevice as sd
mpl.use('TkAgg')
import time

# Options
Audio_On = True  # Audio = False without audio, Audio = True with audio
disturbance_type = 'filtered_white_noise'  # 'sines' or 'filtered_white_noise'

# Audio
file_name = 'sample_dido.mp3'
file_name = 'Suzanne Vega Toms Diner Acapella.mp3'
s, fs = librosa.load(file_name, sr=None, mono=True)
N_samples = 10 * fs
s = s[:N_samples]  # 10s Audio

# Simulation
z = np.zeros(N_samples)

# Disturbance 1
if disturbance_type == 'sines':
    f0 = 80
    df = 40
    t = np.arange(N_samples) / fs
    phi = 2 * np.pi * np.sin(2 * np.pi * 1 * t)
    u = np.arange(N_samples) / N_samples
    z = np.sin(2 * np.pi * (f0 + u * df) * t) + np.sin(2 * np.pi * 2 * (f0 + u * df) * t - phi)

# Disturbance 2
if disturbance_type == 'filtered_white_noise':
    z = 10 * np.random.randn(N_samples)
    b, a = signal.butter(2, 150 / (fs / 2), output='ba')
    z = signal.lfilter(b, a, z)

# Note: z is the interference measured by the reference microphone, e.g., with a microphone outside the earbuds.
# z' is the noise as it is heard inside the headphones and thus by the listener. Here we assume a simple delay
# as a channel model.
damping = 0.5  # Damping
zprime = damping * signal.lfilter([0, 0, 0, 1], 1, z)
d = Audio_On * s + zprime

# Output audio with disturbance removed
#print('Original with interference')
#sd.play(d, fs)  # Use a library like sounddevice to play audio
#sd.wait()

NFIR = 4  # Filter order of FIR filter
b = np.zeros(NFIR + 1)  # FIR filter coefficients
mu_max = 1/(NFIR + 1)
mu = 0.001*mu_max

z = np.pad(z, (NFIR+1,0))
d = np.pad(d, (NFIR+1,0))
s = np.pad(s, (NFIR+1,0))

y = np.zeros_like(z)
e = np.zeros_like(z)

for n in range(NFIR+1, NFIR+1+N_samples):
    x = z[n:n-NFIR-1:-1]
    #print(x)
    y[n] = b@x

    e[n] = d[n] - y[n]

    b += 2*mu*e[n]*x

t = np.arange(N_samples+NFIR+1)/fs
f = np.arange(len(d))*fs/(len(d))

fig, ax = plt.subplots(2,1)
ax[0].set_title('Verauschtes Signal & Original')
ax[0].plot(t, d, label='d[]')
ax[0].plot(t, s, label='s[]')

ax[1].set_title('Gefiltertes Signal & Original')
ax[1].plot(t, s, label='s[]')
ax[1].plot(t, e, label='e[]')
ax[1].set_ylim((np.min(s), np.max(s)))

sd.play(e, fs)  # Use a library like sounddevice to play audio
plt.show()
sd.wait()