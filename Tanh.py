import numpy as np
from Activation import Activation

# Hyperbolic Tangent
class Tanh(Activation):
  def __init__(self):
    tanh = lambda x: np.tanh(x)
    tanh_prime = lambda x: 1 - np.tanh(x) ** 2
    super().__init__(tanh, tanh_prime)