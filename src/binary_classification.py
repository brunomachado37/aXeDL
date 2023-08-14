from axedl.Model import Sequential
from axedl.Layers import Linear
from axedl.Activation import ReLU, Sigmoid
from axedl.Loss import BinaryCrossEntropy
from axedl.Metrics import BinaryAccuracy
from axedl.Optimizer import Adam
from axedl.Data import spiral_data


X_train, y_train = spiral_data(samples = 100, classes = 2)
X_dev, y_dev = spiral_data(samples = 30, classes = 2)

y_train = y_train.reshape(-1, 1)
y_dev = y_dev.reshape(-1, 1)

model = Sequential()
model.add(Linear(2, 64))
model.add(ReLU())
model.add(Linear(64, 1))
model.add(Sigmoid())

model.config(optimizer = Adam(learning_rate = 1e-3, exponential_decay = 5e-7),
             loss = BinaryCrossEntropy(),
             metric = BinaryAccuracy())

model.train(X_train, y_train, epochs = 10000, verbosity = 1000)

# Evaluation
logits = model(X_dev)

loss_dev = BinaryCrossEntropy()(logits, y_dev)
acc_dev = BinaryAccuracy()(logits, y_dev)

print(f'acc_dev: {acc_dev:.3f}, loss_dev: {loss_dev:.3f}')