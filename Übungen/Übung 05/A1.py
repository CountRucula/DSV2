import numpy as np
import matplotlib.pyplot as plt

# Measurement vector
y = np.array([48.54, 47.11, 55.01, 55.15, 49.89, 40.85, 46.72, 50.05, 51.27, 49.95])

# TODO: Kalman Filter









# Visualization
plt.figure(1)
plt.subplot(311)
plt.plot(xhat, '-*')
plt.hlines(50, 0, len(xhat), 'k', ':')
plt.ylabel('Hight xhat')

plt.subplot(312)
plt.plot(P, '-*')
plt.ylabel('Covariance P')

plt.subplot(313)
plt.plot(K, '-*')
plt.ylabel('Kalman gain K')
plt.show()