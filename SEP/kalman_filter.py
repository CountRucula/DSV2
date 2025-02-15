import numpy as np


class KalmanFilter:
    def __init__(self, A, B, C, Q, R, x0, P0):
        self.A = A  # State transition matrix
        self.B = B  # Input matrix
        self.C = C  # Output matrix
        self.Q = Q  # State noise
        self.R = R  # Measurement noise
        self.x = x0  # State estimate
        self.P = P0  # State covariance
        self.K = np.zeros(C.T.shape)

    def measurement_update(self, y):
        S = self.C @ self.P @ self.C.T + self.R
        self.K = self.P @ self.C.T / S
        self.x = self.x + self.K * (y - self.C @ self.x)
        self.P = (np.eye(self.K.shape[0]) - np.outer(self.K, self.C)) @ self.P

    def predict(self, u):
        self.x = self.A @ self.x + np.dot(self.B, u)
        self.P = self.A @ self.P @ self.A.T + self.Q

    def kalman_update(self, y, u):
        self.measurement_update(y)
        self.predict(u)
