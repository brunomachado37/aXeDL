import numpy as np


class Accuracy():
    
    def __call__(self, y_pred, y_true):
        
        preds = np.argmax(y_pred, axis = 1)
        targets = np.argmax(y_true, axis = 1) if len(y_true.shape) == 2 else y_true      # If labels are one-hot

        return np.mean(preds == targets)
    

class BinaryAccuracy():
    
    def __call__(self, logits, y_true):
        preds = (logits > 0.5) * 1
        
        return np.mean(preds == y_true)