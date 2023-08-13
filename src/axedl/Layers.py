import numpy as np
from .Initialization import He


class Layer():
    def __call__(self, X):
        return self.forward(X)
    

class Linear(Layer):
    def __init__(self, input_size, output_size, add_bias = True):
        self.shape = (input_size, output_size)
        self.add_bias = add_bias

        self.W, self.b = He()(input_size, output_size, add_bias)


    def forward(self, X):
        self.X = X
        self.output = np.dot(X, self.W)

        if self.add_bias:
            self.output += self.b

        return self.output
    

    def backward(self, dv):
        self.dW = np.dot(self.X.T, dv)

        if self.add_bias:
            self.db = np.sum(dv, axis = 0, keepdims = True)

        self.dX = np.dot(dv, self.W.T)

        return self.dX


class Dropout(Layer):
    def __init__(self, drop_rate = 0.):
        self.keep_rate = 1. - drop_rate


    def forward(self, X):
        self.dropout_mask = np.random.binomial(1, self.keep_rate, X.shape) / self.keep_rate
        self.output = X * self.dropout_mask 

        return self.output
    

    def backward(self, dv):
        self.dX = dv * self.dropout_mask

        return self.dX