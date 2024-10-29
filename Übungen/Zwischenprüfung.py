#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#%% Aufgabe 1
z = np.array([0.63, -0.11, 0.03, 0.11, -0.41, -0.2, 0.0, 0.23, -0.75, -0.01])

print(np.mean(z))
print(np.mean(z**2))
print(np.var(z, ddof=0))
print(np.std(z, ddof=0))

#%% Aufgabe 3
x = np.array([1,1,1,-1,-1,1,-1], dtype=float)
y = np.array([0.4, 0.5, -0.5, 0.4, 1.3, -0.3, 0.2, -1.0, 0.7, -0.8, -0.6, 0.5])

rxx = np.correlate(x,x, 'full')
print(rxx)

ryx = np.correlate(y,x, 'full')
print(ryx)

D = 3
print(np.dot(x,y[D:D+len(x)]))

#%% Aufgabe 5 a)
A = np.array([[1, 1],
              [1, 3],
              [3, 1],
              [3, 2]], dtype=float)
y = np.array([4.6, 11.3, 9.4, 13.5])

bls = np.linalg.pinv(A)@y
print(bls)
print(np.linalg.norm(A@bls - y)**2)

#%% Aufgabe 5 b)
A = np.array([[1, 1, 1],
              [1, 1, 3],
              [1, 3, 1],
              [1, 3, 2]], dtype=float)

bls = np.linalg.pinv(A)@y
print(bls)
print(np.linalg.norm(A@bls - y)**2)

#%% Aufgabe 6 b)
A = np.array([[-0.9, -0.9],
              [0.8,  -0.9],
              [0.3,  0.8],
              [-0.6, 0.3]], dtype=float)

d = np.array([0.0, 1.7, -0.5, -0.9])

bls = np.linalg.pinv(A)@d
print(bls)
print(d-A@bls)
print(np.linalg.norm(A@bls - d)**2)