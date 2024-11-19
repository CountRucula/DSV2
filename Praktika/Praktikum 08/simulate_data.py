import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
Ts = 0.5  # [s]
N = 30  # samples
ACC_CONST = 20  # [m/s^2]
GRAVITY_ACCEL = 9.81  # [m/s^2]
sigma_u = 0 # 0.1  # [m/s^2]
sigma_height = 50  # [m]

# Initialization
v_true = np.zeros(N)
y_true = np.zeros(N)
y_meas = np.zeros(N)

# Simulate trajectory
u_control = (ACC_CONST + sigma_u * np.random.randn(N)) - GRAVITY_ACCEL

for n in range(N-1):
    y_true[n+1] = y_true[n] + v_true[n]*Ts + 0.5*u_control[n]*(Ts**2)
    v_true[n+1] = v_true[n] + u_control[n]*Ts

y_meas = y_true + np.random.randn(N) * sigma_height
t = np.arange(N)

# Visualize
plt.figure(1)
plt.plot(t, y_true)
plt.plot(t, y_meas,'x')
plt.xlabel("Time [s]")
plt.ylabel("Height [m]")
plt.legend(["True", "Measured"])
plt.grid(True)
plt.show()

# Save data
np.savez("measurement_data", y_meas=y_meas, Ts=Ts, sigma_u=sigma_u, sigma_height=sigma_height, ACC_CONST=ACC_CONST, GRAVITY_ACCEL=GRAVITY_ACCEL)
np.savez("groundtruth_data", y_true=y_true, v_true=v_true, Ts=Ts)