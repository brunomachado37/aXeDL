import numpy as np


class Naive():
    def __init__(self, alpha = 0.01):
        self.alpha = alpha

    def __call__(self, input_size, output_size, add_bias):
        W = self.alpha * np.random.randn(input_size, output_size)
        b = np.zeros((1 , output_size)) if add_bias else None

        return W, b


class Xavier():
    def __call__(self, input_size, output_size, add_bias):
        W = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        b = np.zeros((1 , output_size)) if add_bias else None

        return W, b


class He():
    def __call__(self, input_size, output_size, add_bias):
        W = np.random.randn(input_size, output_size) / np.sqrt(input_size / 2)
        b = np.zeros((1 , output_size)) if add_bias else None

        return W, b


class Uniform():
    def __call__(self, input_size, output_size, add_bias):
        stdv = 1. / np.sqrt(input_size)

        W = np.random.uniform(-stdv, stdv, (input_size, output_size))
        b = np.random.uniform(-stdv, stdv, (1, output_size)) if add_bias else None

        return W, b