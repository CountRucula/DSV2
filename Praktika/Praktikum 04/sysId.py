import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

# Parameters
fs = 48e3  # Sampling frequency
Niter = 2000  # TODO: Number of iterations
N_FIR = 64  # 
mu_max = 1 / (N_FIR + 1)  # Max step size when E[x^2]=1

# TODO: Unknown system H(z) that is to be identified
h = np.zeros(N_FIR+1)

# a)
#h[0] = 1

# b)
#h[0] = 0.5
#h[1] = 0.5

# c)
h = scipy.signal.firwin(N_FIR+1, fs=fs, cutoff=8000)


# LMS parameter initialization
mu = 0.5 * mu_max  # Try mu=0.1*mu_max (rule of thumb)
b = np.zeros(N_FIR + 1)  # Initial coefficients of adaptive filter
xn_vec = np.zeros(N_FIR + 1, dtype=float)  # Samples in FIR-Tapped-Delay-Line
e = np.zeros(Niter)

# TODO: Iterative steps of LMS-Algorithm
d = np.zeros(Niter)
y = np.zeros(Niter)

plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(3,1)

ax1.set_title('Filter-Koeffizienten')
ax1.grid(True)
ax1.plot(h, '.')

ax2.set_title('gelernte Koeffizienten')
ax2.grid(True)
ax2.set_ylim((-0.1, 0.35))
line1, = ax2.plot(b, '.')

ax3.set_title('Error-Quadrat')
ax3.grid(True)
ax3.set_ylim((0, 1))
line2, = ax3.plot(e**2)

plt.tight_layout()

for n in range(Niter):
    # gen new value
    xn_vec = np.concatenate((xn_vec, [np.random.randn()]),axis=0)

    x = xn_vec[-1:-N_FIR-2:-1]

    # gen d[n]
    d[n] = h@x

    # calc y[n]
    y[n] = b@x

    #print(f'x    = {x}')
    #print(f'h    = {h}')
    #print(f'b    = {b}')
    #print(f'd[n] = {d[n]}')
    #print(f'y[n] = {y[n]}')

    # calc e[n]
    e[n] = d[n] - y[n]

    # calc b[n+1]
    #for k in range(N_FIR):
        #b[k] = b[k] + 2*mu*e[n]*x[k]
    b += 2*mu*e[n]*x

    line1.set_ydata(b)
    line2.set_ydata(e**2)

    plt.pause(0.003)
    
    
print(b)

plt.ioff()
plt.show()
