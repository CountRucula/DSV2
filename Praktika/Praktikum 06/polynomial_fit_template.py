#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
N = 101
x = -1 + 0.02*np.arange(1, N+1)  # Stützstellen
d = np.sin(np.pi*x)

y = np.zeros(d.shape)

f = [
    lambda x: x**5,
    lambda x: x**4,
    lambda x: x**3,
    lambda x: x**2,
    lambda x: x,
    lambda x: 1,
]

#%%
A = np.array([[f(x) for f in f] for x in x])

bls = np.linalg.pinv(A)@d
y = A@bls

b_po = np.polyfit(x,d,deg=5)
y_po = A@b_po

#%%

# Visualize results
plt.figure(1)
plt.subplot(211)
plt.plot(x, d, x, y, x, y_po)
plt.legend(["f(x)", "Least Squares" ,"Polyfit p(x)"])
plt.grid(True)

plt.subplot(212)
plt.plot(x, d-y)
plt.ylabel("Error")
plt.grid(True)
plt.show()