import numpy as np


class ExtendedKalmanFilter:
    def __init__(self, f, F, h, H, Q, R, x0, P0):
        self.f = f  
        self.F = F  
        self.h = h  
        self.H = H

        self.Q = Q  # State noise
        self.R = R  # Measurement noise
        self.x = x0  # State estimate
        self.P = P0  # State covariance

    def measurement_update(self, y):
        H = self.H(self.x)

        S = H @ self.P @ H.T + self.R
        self.K = self.P @ H.T @ np.linalg.pinv(S)

        self.x = self.x + self.K @ (y - self.h(self.x))
        self.P = (np.eye(self.K.shape[0]) - self.K @ H) @ self.P

        #S = self.C @ self.P @ self.C.T + self.R
        #self.K = self.P @ self.C.T / S
        #self.x = self.x + self.K * (y - self.C @ self.x)
        #self.P = (np.eye(self.K.shape[0]) - np.outer(self.K, self.C)) @ self.P

    def predict(self, u):
        self.x = self.f(self.x, u)

        F = self.F(self.x, u)
        self.P = F @ self.P @ F.T + self.Q

        #self.x = self.A @ self.x + np.dot(self.B, u)
        #self.P = self.A @ self.P @ self.A.T + self.Q

    def kalman_update(self, y, u):
        self.measurement_update(y)
        self.predict(u)
