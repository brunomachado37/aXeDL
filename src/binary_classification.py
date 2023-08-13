from axedl.Layers import Linear
from axedl.Activation import ReLU, Sigmoid
from axedl.Loss import BinaryCrossEntropy
from axedl.Metrics import BinaryAccuracy
from axedl.Optmizer import Adam
from axedl.Data import spiral_data


X_train, y_train = spiral_data(samples = 100, classes = 2)
X_dev, y_dev = spiral_data(samples = 100, classes = 2)

y_train = y_train.reshape(-1, 1)
y_dev = y_dev.reshape(-1, 1)

linear_1 = Linear(2, 64)
activation_1 = ReLU()
linear_2 = Linear(64, 1)
activation_2 = Sigmoid()

loss_fct = BinaryCrossEntropy()
metric = BinaryAccuracy()
optimizer = Adam(learning_rate = 1e-3, exponential_decay = 5e-7)

for epoch in range(10001):
    # Forward pass
    Z_1 = linear_1(X_train)
    A_1 = activation_1(Z_1)
    Z_2 = linear_2(A_1)
    logits = activation_2(Z_2)

    loss = loss_fct(logits, y_train)
    acc = metric(logits, y_train)
    
    if not epoch % 100:
        print(f'epoch: {epoch}, acc: {acc:.3f}, loss: {loss:.3f}, lr: {optimizer.learning_rate:.3f}')

    # Backward pass
    dv = loss_fct.backward(logits, y_train)
    dv = activation_2.backward(dv)
    dv = linear_2.backward(dv)
    dv = activation_1.backward(dv)
    dv = linear_1.backward(dv)

    # Update
    optimizer((linear_1, linear_2))


# Evaluation
Z_1 = linear_1(X_dev)
A_1 = activation_1(Z_1)
Z_2 = linear_2(A_1)
logits = activation_2(Z_2)

loss_dev = loss_fct(logits, y_dev)
acc_dev = metric(logits, y_dev)

print(f'acc_dev: {acc_dev:.3f}, loss_dev: {loss_dev:.3f}')