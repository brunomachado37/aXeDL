import numpy as np
from .Activation import Softmax


class Loss():

    def __init__(self, reduction = 'mean'):
        self.reduction = reduction

    def __call__(self, y_pred, y_true):
        losses = self.forward(y_pred, y_true)
        cost = self.reduce(losses)

        return cost

    def reduce(self, losses):
        if self.reduction == 'mean':
            return np.mean(losses)
        
        elif self.reduction == 'max':
            return np.max(losses)
        
        else:
            return losses
        

class CategoricalCrossEntropy(Loss):

    def __init__(self, reduction = 'mean'):
        super().__init__(reduction)

        self.activation = Softmax()
        self.loss = CategoricalCrossEntropyWithoutSoftmax() 


    def forward(self, logits, y_true):
        y_pred = self.activation.forward(logits)

        return self.loss.forward(y_pred, y_true)
    

    def backward(self, dv, y_true):
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)        # Convert labels into categorical

        self.dX = dv.copy()
        self.dX[range(len(dv)), y_true] -= 1
        self.dX /= len(dv)                              # Normalize gradient

        return self.dX
    

class CategoricalCrossEntropyWithoutSoftmax(Loss):      #TODO: Rename based on PyTorch

    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)        # Clip to prevent division per zero

        if len(y_true.shape) == 1:                      # Categorical Labels
            conf = y_pred[range(len(y_pred)), y_true]

        elif len(y_true.shape) == 2:                    # One-hot labels
            conf = np.sum(y_pred * y_true, axis = 1)

        else:
            raise ValueError("y_true must contain categorical labels (integers) or one-hot labels")


        return -np.log(conf)
    

    def backward(self, dv, y_true):
        if len(y_true.shape) == 1:
            y_true = np.eye(len(dv[0]))[y_true]         # Convert labels into one-hot

        self.dX = -y_true / dv
        self.dX /= len(dv)                              # Gradient normalization

        return self.dX
    

class BinaryCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)        # Clip to prevent division per zero

        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis = -1)
    

    def backward(self, dv, y_true):
        dv = np.clip(dv, 1e-7, 1 - 1e-7)                # Clip to prevent division per zero

        self.dX = -(y_true / dv - (1 - y_true) / (1 - dv)) / len(dv[0])
        self.dX /= len(dv)                              # Gradient normalization

        return self.dX