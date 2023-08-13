from axedl.Layers import Linear, Dropout
from axedl.Activation import ReLU
from axedl.Loss import CategoricalCrossEntropy
from axedl.Metrics import Accuracy
from axedl.Optmizer import Adam
from axedl.Data import spiral_data


X_train, y_train = spiral_data(samples = 100, classes = 3)
X_dev, y_dev = spiral_data(samples = 100, classes = 3)

linear_1 = Linear(2, 128)
activation_1 = ReLU()
dropout_1 = Dropout(0.2)
linear_2 = Linear(128, 3)

loss_fct = CategoricalCrossEntropy()
metric = Accuracy()
optimizer = Adam(learning_rate = 0.07, exponential_decay = 5e-7)

for epoch in range(5001):
    # Forward pass
    Z_1 = linear_1(X_train)
    A_1 = activation_1(Z_1)
    D_1 = dropout_1(A_1)
    logits = linear_2(D_1)

    loss = loss_fct(logits, y_train)
    acc = metric(loss_fct.activation.output, y_train)

    if not epoch % 100:
        print(f'epoch: {epoch}, acc: {acc:.3f}, loss: {loss:.3f}, lr: {optimizer.learning_rate:.3f}')

    # Backward pass
    dv = loss_fct.backward(loss_fct.activation.output, y_train)
    dv = linear_2.backward(dv)
    dv = dropout_1.backward(dv)
    dv = activation_1.backward(dv)
    dv = linear_1.backward(dv)

    # Update
    optimizer((linear_1, linear_2))


# Evaluation
Z_1 = linear_1(X_dev)
A_1 = activation_1(Z_1)
logits = linear_2(A_1)

loss_dev = loss_fct(logits, y_dev)
acc_dev = metric(loss_fct.activation.output, y_dev)

print(f'acc_dev: {acc_dev:.3f}, loss_dev: {loss_dev:.3f}')