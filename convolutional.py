from Layer import Layer
import numpy as np
from scipy import signal

class Convolutional(Layer):
    def __int__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernel_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.rand(*self.kernel_shape)
        self.biases = np.random.randn(*self.output_shape)
    
    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output += signal.correlate2d(self.input[j], self.kernels[i, j], "valid") #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate2d.html
        return self.output
    
    def backward(self, output_graident, learning_rate):
        kernels_gradient = np.zeros(self.kernel_shape)
        input_graident = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i ,j] = signal.correlate2d(self.input[j], output_graident[i], "valid")
                input_graident[j] += signal.convolve2d(output_graident[i], self.kernels[i, j], "full")
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_graident
        return input_graident