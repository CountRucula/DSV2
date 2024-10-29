#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#%%
x = np.array([-2.0, -1.0, -0.7, 1.3])
d = np.array([-4.1, -1.9, -1.5, 2.5])

#%% Aufgabe 1
print(1/(x.T@x)*x.T@d)

#%% Aufgabe 2
z = np.array([0.299, 0.289, 1.244, 0.911, 0.902, 1.71, 1.146, 1.099, 1.794, 0.598])

print(np.mean(z))
print(np.mean(z**2))
print(np.var(z, ddof=0))
print(np.std(z, ddof=0))

#%% Aufgabe 2 c)
