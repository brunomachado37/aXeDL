from axedl.Model import Sequential
from axedl.Layers import Linear, Dropout
from axedl.Activation import ReLU
from axedl.Loss import CategoricalCrossEntropy
from axedl.Metrics import CategoricalAccuracy
from axedl.Optimizer import Adam
from axedl.Data import spiral_data


X_train, y_train = spiral_data(samples = 1000, classes = 3)
X_dev, y_dev = spiral_data(samples = 100, classes = 3)

model = Sequential()
model.add(Linear(2, 512))
model.add(ReLU())
model.add(Dropout(0.1))
model.add(Linear(512, 3))

model.config(optimizer = Adam(learning_rate = 0.07, exponential_decay = 5e-7),
             loss = CategoricalCrossEntropy(),
             metric = CategoricalAccuracy())

model.train(X_train, y_train, epochs = 10000, verbosity = 1000)


# Evaluation
logits = model(X_dev)

loss_fct = CategoricalCrossEntropy()
loss_dev = loss_fct(logits, y_dev)
acc_dev = CategoricalAccuracy()(loss_fct.activation.output, y_dev)

print(f'acc_dev: {acc_dev:.3f}, loss_dev: {loss_dev:.3f}')