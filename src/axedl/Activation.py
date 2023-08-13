import numpy as np
from .Layers import Layer


class ReLU(Layer):   
    def forward(self, X):
        self.X = X
        self.output = np.maximum(0, X)

        return self.output
    
    def backward(self, dv):
        self.dX = dv.copy()
        self.dX[self.X <= 0] = 0

        return self.dX
    

class Sigmoid(Layer):
    def forward(self, X):
        self.output = 1 / (1 + np.exp(-X))

        return self.output
    
    def backward(self, dv):
        self.dX = dv * self.output * (1 - self.output)

        return self.dX


class Softmax(Layer):
    def forward(self, X):
        Z = np.exp(X - np.max(X, axis = 1, keepdims = True))    # Subtract the max to clip the input values between some negative number and zero, to prevent overflow with the exponential
        self.output = Z / np.sum(Z, axis = 1, keepdims = True)
        
        return self.output
    
    def backward(self, dv):
        self.dX = np.empty_like(dv)

        for idx, (_output, _dv) in enumerate(zip(self.output, dv)):
            _output = _output.reshape(-1, 1)
            
            jacobian_matrix = np.diagflat(_output) - np.dot(_output, _output.T)
            self.dX[idx] = np.dot(jacobian_matrix, _dv)

        return self.dX


class Step(Layer):
    def forward(self, X):
        self.output = np.fmax(0, X)
        self.output[self.output != 0] = 1

        return self.output