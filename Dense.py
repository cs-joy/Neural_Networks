import numpy as np
from Layer import Layer

# Dense Layer/ Fully Connected Layer
class Dense(Layer):
  def __init__(self, input_size, output_size): # input_size= number of neurons in input layer, and output_size= number of neurons in output layer
    self.weights = np.random.randn(output_size, input_size) # input_size = i and output_size=j, so shape will be (jxi)
    self.bias = np.random.randn(output_size, 1) # output_size = j, so shape will be (jx1)
  
  def forward(self, input): # calculate Y = W . X + B
    self.input = input
    return np.dot(self.weights, self.input) + self.bias # W . X + B, where W = self.weights and X = self.input and B = self.bias

  def backward(self, output_gradient, learning_rate): # output_gradient = δE/δY
    weights_graident = np.dot(output_gradient, learning_rate) # δE/δW = δE/δY . X^t, where t means transpose and X^t means transpose of X matrix
    #bias_gradient = output_gradient # δE/δB = δE/δY, since derivative of the error respect to bias(δE/δB) is equal to derivative of the error respect to output(δE/δY), hence we can ignore this line of code and thus i comment this code
    self.weights -= learning_rate * weights_graident # updating weights parameter, W
    self.bias-= learning_rate * output_gradient # updating bias parameter, B
    return np.dot(self.weights.T, output_gradient) # calculate the derivative of the error respect to the input(X), δE/δX = W^t . δE/δY, where W^t means transpose matrix of W
