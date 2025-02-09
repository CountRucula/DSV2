import numpy as np
import matplotlib.pyplot as plt

def zscore(X):
    meanX = np.mean(X, axis=0)
    stdX = np.std(X, axis=0)
    X = (X - meanX) / stdX
    return X, meanX, stdX

def myActivation(X):
    return 1*(1 + np.exp(-X))

def myDiffSigmoid(X):
    return myActivation(X) * (1 - myActivation(X))

def myInference(W1, W2, X):
    # Layer 1
    v1 = W1@X
    print(v1)
    y1 = myActivation(v1)

    # Layer 2 (Output Layer)
    v2 = y1@W2
    y2 = myActivation(v2)    
    
    return y1, y2, v1, v2

def myLoss(y, d):
    return - d/y + (1-d)/(1-y)

# Initialization
T = 4

# TODO (Training data): X =
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float).T

X, meanX, stdX = zscore(X.T)
X = np.vstack((X.T, np.ones((1, T))))
D = np.array([0, 1, 1, 0]).reshape(1, -1)
N = 4

alpha = 0.9  # Learning rate
MaxEpoch = 2000
Output = np.zeros((T, MaxEpoch))

# Xavier weight initialization
# TODO: Try other initializations
in_dim, out_dim = 3, 4
W1 = (2 * np.random.rand(out_dim, in_dim) - 1) * np.sqrt(6) / np.sqrt(in_dim + out_dim)
in_dim, out_dim = 4, 1
W2 = (2 * np.random.rand(in_dim, out_dim) - 1) * np.sqrt(6) / np.sqrt(in_dim + out_dim)

for epoch in range(MaxEpoch):
    # TODO: Forward pass
    Y1, Y2, V1, V2 = myInference(W1, W2, X)

    # TODO: Backpropagation

    # Layer 2
    e2 = myLoss(Y2, D)
    delta_2 = myDiffSigmoid(V2) * e2
    dW2 = np.dot(delta_2, Y1.T) / N

    # Layer 1
    e1 = np.dot(delta_2, W2)
    delta_1 = myDiffSigmoid(V1) * e1
    dW1 = np.dot(delta_1, X.T) / N

    # Weight update
    W2 -= alpha * dW2
    W1 -= alpha * dW1

    # Calculate output with "new" weights
    _, Y2, _, _ = myInference(W1, W2, X)
    Output[:, epoch] = Y2.flatten()

# Print predictions at final iteration
print("[D Y2]")
print(np.hstack((D.T, Y2.T)))

plt.plot(range(1, MaxEpoch + 1), Output.T, '-x')
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('Output')
plt.legend(['y1', 'y2', 'y3', 'y4'])
plt.show()