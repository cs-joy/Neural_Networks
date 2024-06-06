import numpy as np
from activations import Activation

# Hyperbolic Tangent
class Tanh(Activation):
  def __init__(self):
    tanh = lambda x: np.tanh(x)
    tanh_prime = lambda x: 1 - np.tanh(x) ** 2
    super().__init__(tanh, tanh_prime)

# Sigmoid
class Sigmoid(Activation):
  def __init__(self):
    def sigmoid(x):
      return 1 / (1 + np.exp(-x))
    
    def sigmoid_prime(x):
      s = sigmoid(x)
      return s * (1 - s)
    
    super().__init__(sigmoid, sigmoid_prime)