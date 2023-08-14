from .Layers import Layer
from .Optimizer import Optimizer
from .Loss import Loss

class Sequential:
    def __init__(self):
        self.layers = []


    def __call__(self, X):
        return self.forward(X)


    def add(self, layer):
        if isinstance(layer, Layer):
            self.layers.append(layer)
        
        else:
            raise ValueError('Only objects of type Layer can be added to a Sequential model')
        

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        
        return X


    def backward(self, logits, y):
        dv = self.loss.backward(logits, y)

        for layer in reversed(self.layers):
            dv = layer.backward(dv)
        
        return dv
    

    def config(self, optimizer, loss, metric = None):
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError(f'{type(optimizer)} object cannot be set as Optimizer')
        
        if isinstance(loss, Loss):
            self.loss = loss
        else:
            raise ValueError(f'{type(loss)} object cannot be set as Loss function')
        
        self.metric = metric

    
    def train(self, X, y, epochs = 1, verbosity = 1, X_dev = None, y_dev = None):
        if not (hasattr(self, 'optimizer') and hasattr(self, 'loss')):
            raise AttributeError('Optimizer or Loss function not defined, use config method first')
        
        for epoch in range(1, epochs + 1):
            logits = self.forward(X)
            loss = self.loss(logits, y)
            metric = self.metric(logits, y) if self.metric else None

            self.backward(logits, y)
            self.optimizer(self.layers)

            self._display_info(epoch, verbosity, loss, metric)
            

    def _display_info(self, epoch, verbosity, loss, metric):        
        if not epoch % verbosity:
            log = f'epoch: {epoch} | loss: {loss:.3f} | lr: {self.optimizer.learning_rate:.3f}'
            
            if self.metric:
                log += f" | {self.metric.name}: {metric:.3f}"

            print(log)

