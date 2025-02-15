import numpy as np
import matplotlib.pyplot as plt

# System model
A = 1
B = 0
C = 1
Q = 0
sigma_m = 0.1
R = sigma_m**2

# true values
N = 30
x = 50 + 0.3*np.arange(N)/N  # T steigt um 0.01°C pro Messung

# Measurement
# y = x + sigma_m*np.random.randn(N);

y = np.array([49.8807, 50.0747, 49.9846, 50.0346, 49.9607, 49.8949, 50.0772,
              50.0638, 50.1999, 50.1702, 50.2053, 50.0351, 50.0264, 50.0031,
              50.1898, 50.4289, 50.2328, 50.0927, 50.2637, 50.0772, 50.0576,
              50.2817, 50.1422, 50.2616, 50.3807, 50.2901, 50.3530, 50.1094,
              50.3462, 50.5039])

# TODO: Kalman filter




# Plotting
plt.subplot(3, 1, 1)
plt.plot(np.arange(N), xhat, '-o', np.arange(N), y, '-x', np.arange(N), x, '-g')
plt.grid(True)
plt.xlabel('Messung')
plt.legend(['estimation', 'measurement', 'true values'], loc='upper left')
plt.axis([0, N, 49.8, 50.5])

plt.subplot(3, 1, 2)
plt.plot(np.arange(N), P, '-o')
plt.grid(True)
plt.xlabel('Messung')
plt.ylabel('Unsicherheit P[n]')
plt.axis([0, N, 0, 0.01])

plt.subplot(3, 1, 3)
plt.plot(np.arange(N), K, '-o')
plt.grid(True)
plt.xlabel('Messung')
plt.ylabel('Kalman Gain K[n]')
plt.axis([0, N, 0, 1])

plt.show()
