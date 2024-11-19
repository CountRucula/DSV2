import numpy as np
import matplotlib.pyplot as plt
from kalman_filter_with_input import KalmanFilter

# Load simulated measurement data
meas_data = np.load("measurement_data.npz")
y_meas = meas_data['y_meas']
Ts = meas_data['Ts']
ACC_CONST = meas_data['ACC_CONST']
g0 = meas_data['GRAVITY_ACCEL']
N = len(y_meas)

# State space model
A = 0 # TODO: Beispiel 3 aus der Vorlesung
B = 0 # TODO: Beispiel 3
C = np.array([1, 0])
Q = (meas_data['sigma_u']**2) * np.array([[0.25*Ts**4, 0.5*Ts**3], [0.5*Ts**3, Ts**2]])
R = meas_data['sigma_height']**2

# Initialize Kalman filter and state estimates
xhat = np.zeros((2,N))
P = np.zeros((2,2,N))
u = np.ones(N)*(ACC_CONST - g0)
Kgain = np.zeros((2,N))
x0 = np.array([0,0])
P0 = 1e3*np.eye(2)
kf = KalmanFilter(A, B, C, Q, R, x0, P0)

for k in range(N):

    # TODO
    xhat[:,k] =
    P[:,:,k] =
    Kgain[:,k] =


# Visualize results
groundtruth_data = np.load("groundtruth_data.npz")
t = np.arange(N)*Ts

plt.figure(1)
plt.subplot(411)
plt.plot(t, y_meas, 'xr')
plt.plot(t, groundtruth_data['y_true'], '-g')
plt.plot(t, xhat[0,:], '-bo')
plt.legend([ 'Measurement', 'True', 'Estimate'])
plt.ylabel("Flughöhe [m]")
plt.grid(True)

plt.subplot(412)
plt.plot(t, groundtruth_data['v_true'], '-g')
plt.plot(t, xhat[1,:], '-bo')
plt.legend(['True', 'Estimate'])
plt.ylabel("Geschwindigkeit [m/s]")
plt.grid(True)

plt.subplot(413)
plt.plot(t, P[0,0,:], '-g')
plt.plot(t, P[1,1,:], '-r')
plt.ylabel("State covariance")
plt.legend(['P11', 'P22'])
plt.grid(True)

plt.subplot(414)
plt.plot(t, Kgain[0,:], '-g')
plt.plot(t, Kgain[1,:], '-r')
plt.xlabel("Zeit [s]")
plt.ylabel("Kalman gain")
plt.legend(['K11', 'K12'])
plt.grid(True)

plt.show()