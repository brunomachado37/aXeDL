import numpy as np


class Optimizer():

    def __init__(self, learning_rate = 1e-3, exponential_decay = 0.):
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate

        self.exponential_decay = exponential_decay

        self.iter = 0


    def __call__(self, layers):
        if self.exponential_decay:
            self.learning_rate = self.initial_learning_rate / (1. + self.exponential_decay * self.iter)

        for layer in layers:
            if hasattr(layer, 'W'):
                self._update(layer)

        self.iter += 1


class Adam(Optimizer):

    def __init__(self, learning_rate = 1e-3, exponential_decay = 0., eps = 1e-7, beta_1 = 0.9, beta_2 = 0.999):
        super().__init__(learning_rate, exponential_decay)

        self.eps = eps
        self.beta_1 = beta_1
        self.beta_2 = beta_2


    def _update(self, layer):
        if not hasattr(layer, 'W_cache'):
            layer.W_cache = np.zeros_like(layer.W)
            layer.b_cache = np.zeros_like(layer.b)

            layer.W_momentum = np.zeros_like(layer.W)
            layer.b_momentum = np.zeros_like(layer.b)

        layer.W_momentum = self.beta_1 * layer.W_momentum + (1 - self.beta_1) * layer.dW
        layer.b_momentum = self.beta_1 * layer.b_momentum + (1 - self.beta_1) * layer.db
        W_momentum = layer.W_momentum / (1 - self.beta_1 ** (self.iter + 1))
        b_momentum = layer.b_momentum / (1 - self.beta_1 ** (self.iter + 1))

        layer.W_cache = self.beta_2 * layer.W_cache + (1 - self.beta_2) * layer.dW ** 2
        layer.b_cache = self.beta_2 * layer.b_cache + (1 - self.beta_2) * layer.db ** 2
        W_cache = layer.W_cache / (1 - self.beta_2 ** (self.iter + 1))
        b_cache = layer.b_cache / (1 - self.beta_2 ** (self.iter + 1))

        layer.W -= self.learning_rate * W_momentum / (np.sqrt(W_cache) + self.eps)
        layer.b -= self.learning_rate * b_momentum / (np.sqrt(b_cache) + self.eps)


class RMSProp(Optimizer):

    def __init__(self, learning_rate = 1e-3, exponential_decay = 0., eps = 1e-7, rho = 0.999):
        super().__init__(learning_rate, exponential_decay)

        self.eps = eps
        self.rho = rho


    def _update(self, layer):
        if not hasattr(layer, 'W_cache'):
            layer.W_cache = np.zeros_like(layer.W)
            layer.b_cache = np.zeros_like(layer.b)

        layer.W_cache = self.rho * layer.W_cache + (1 - self.rho) * layer.dW ** 2
        layer.b_cache = self.rho * layer.b_cache + (1 - self.rho) * layer.db ** 2

        layer.W -= self.learning_rate * layer.dW / (np.sqrt(layer.W_cache) + self.eps)
        layer.b -= self.learning_rate * layer.db / (np.sqrt(layer.b_cache) + self.eps)


class AdaGrad(Optimizer):

    def __init__(self, learning_rate = 1e-3, exponential_decay = 0., eps = 1e-7):
        super().__init__(learning_rate, exponential_decay)
        self.eps = eps


    def _update(self, layer):
        if not hasattr(layer, 'W_cache'):
            layer.W_cache = np.zeros_like(layer.W)
            layer.b_cache = np.zeros_like(layer.b)

        layer.W_cache += layer.dW ** 2
        layer.b_cache += layer.db ** 2

        layer.W -= self.learning_rate * layer.dW / (np.sqrt(layer.W_cache) + self.eps)
        layer.b -= self.learning_rate * layer.db / (np.sqrt(layer.b_cache) + self.eps)


class StochasticGradientDescent(Optimizer):

    def __init__(self, learning_rate = 1e-3, exponential_decay = 0., momentum = 0.):
        super().__init__(learning_rate, exponential_decay)

        self.momentum = momentum


    def _update(self, layer):
        if self.momentum:
            if not hasattr(layer, 'W_momentum'):
                layer.W_momentum = np.zeros_like(layer.W)
                layer.b_momentum = np.zeros_like(layer.b)

            layer.W_momentum = self.momentum * layer.W_momentum - self.learning_rate * layer.dW
            layer.b_momentum = self.momentum * layer.b_momentum - self.learning_rate * layer.db

            layer.W += layer.W_momentum
            layer.b += layer.b_momentum

        else:
            layer.W -= self.learning_rate * layer.dW
            layer.b -= self.learning_rate * layer.db