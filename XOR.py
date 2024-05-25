import numpy as np
from Dense import Dense
from Tanh import Tanh
from Losses import mse, mse_prime

# XOR
X = np.reshape([[0, 0], [0,1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1] [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

epochs = 10000
learning_rate = 0.1

# train
for e in range(epochs):
  error = 0
  for x, y in zip(X, Y):
    #forward
    output = x
    for layer in network:
      output = layer.forward(output)
    
    error += mse(y, output)

    # backward
    grad = mse_prime(y, output)
    for layer in reversed(network):
      grad = layer.backward(grad, learning_rate)
    
  error /= len(X)
  print(f'{e+1}/{epochs}, error={error}')