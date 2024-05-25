from Layer import Layer
import numpy as np

# Activation Layer
class Activation(Layer):
  def __init__(self, activation, activation_prime):
    self.activation = activation
    self.activation_prime = activation_prime
  
  def forward(self, input):
    self.input = input
    return self.activation(self.input) # Y = f(X), where f is a activation function and X is the input
  
  def backward(self, output_gradient, learning_rate):
    return np.multiply(self.output_gradient, self.activation_prime(self.input)) # δE/δX = δE/δY . f'(X) where f' = activation_prime, here