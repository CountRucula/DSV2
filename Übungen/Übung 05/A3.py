import numpy as np
import matplotlib.pyplot as plt

# System-Modell
Ts = 0.25
A = np.array([[1, Ts, 0.5*Ts**2], [0, 1, Ts], [0, 0, 1]])
C = np.array([[1, 0, 0], [0, 0, 1]])
sigma_h = 0  # Genauigkeit Höhen-Messwerte
sigma_a = 0 # Genauigkeit Beschleunigungsmessung
R = np.diag([sigma_h**2, sigma_a**2])
Q = 0*np.eye(3)  # Prozessrauschen

N = 30  # Anzahl Iterationen

# true value for state
x = np.zeros(N)
v = np.zeros(N)
a = 30*np.ones(N)  # konstante Beschleunigung
#a = 30*np.ones(N) + np.arange(0,3*N-1,3)  # Beschleunigung nimmt zu

for n in range(1, N):
    x[n] = x[n-1] + Ts*v[n-1] + 0.5*(Ts**2)*a[n-1]
    v[n] = v[n-1] + Ts*a[n-1]

# 30 noisy measurements
y = np.zeros((2, N))
y[0, :] = np.array([-32.4, -11.1, 18, 22.9, 19.5, 28.5, 46.5, 68.9, 48.2, 56.1, 90.5, 104.9, 140.9, 148, 187.6, 209.2, 244.6, 276.4, 323.5, 357.3, 357.4, 398.3, 446.7, 465.1, 529.4, 570.4, 636.8, 693.3, 707.3, 748.5])
y[1, :] = np.array([39.72, 40.02, 39.97, 39.81, 39.75, 39.6, 39.77, 39.83, 39.73, 39.87, 39.81, 39.92, 39.78, 39.98, 39.76, 39.86, 39.61, 39.86, 39.74, 39.87, 39.63, 39.67, 39.96, 39.8, 39.89, 39.85, 39.9, 39.81, 39.81, 39.68]) - 9.8

#y[0,:] = x + sigma_h*np.random.randn(1,N)
#y[1,:] = a + sigma_a*np.random.randn(1,N)

# Kalman Filter Initialization
xhat_0 = np.array([0, 0, 9.8])  # estimated state
P_0 = 500*np.eye(3)  # estimate uncertainty
xhat_m1 = np.zeros((3, N+1))
P_m1 = np.zeros((3, 3, N+1))
xhat_m1[:, 0] = np.dot(A, xhat_0)
P_m1[:, :, 0] = np.dot(np.dot(A, P_0), A.T) + Q

# Kalman Filter Iterations
K = np.zeros((3, 2, N))
xhat = np.zeros((3, N))
P = np.zeros((3, 3, N))
for n in range(N):
    K[:, :, n] = np.dot(np.dot(P_m1[:, :, n], C.T), np.linalg.inv(np.dot(np.dot(C, P_m1[:, :, n]), C.T) + R))  # Kalman Gain
    xhat[:, n] = xhat_m1[:, n] + np.dot(K[:, :, n], (y[:, n] - np.dot(C, xhat_m1[:, n])))
    P[:, :, n] = np.dot(np.eye(3) - np.dot(K[:, :, n], C), P_m1[:, :, n])
    xhat_m1[:, n+1] = np.dot(A, xhat[:, n])
    P_m1[:, :, n+1] = np.dot(np.dot(A, P[:, :, n]), A.T) + Q

t = np.arange(30)/4
plt.subplot(2, 1, 1)
plt.plot(t, xhat[0, :], '-o', t, y[0, :], '-x', t, x, '-g')
plt.grid(True)
plt.xlabel('Zeit t / s')
plt.ylabel('Flughöhe')
plt.legend(['xhat_1[n] (Schätzung Höhe)', 'y_1[n] (Messung Höhe)', 'echte Höhe'])

kh = K[0, 0, :]
ka = K[2, 1, :]

plt.subplot(2, 1, 2)
plt.plot(t, kh, '-o', t, ka, '-x')
plt.grid(True)
plt.xlabel('Zeit t / s')
plt.ylabel('Kalman gain')
plt.legend(['K_11[n] (Höhe)', 'K_32[n] (Beschleunigung)'])
plt.axis([0, 8, 0, 1])
plt.show()
