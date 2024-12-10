from EDF import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define constants hyperparamters
CLASS1_SIZE = 100
CLASS2_SIZE = 100
N_FEATURES = 2
N_OUTPUT = 1
LEARNING_RATE = 0.1
EPOCHS = 100
TEST_SIZE = 0.25

# Define the means and covariances of the two components
MEAN1 = np.array([1, 2])
COV1 = np.array([[1, 0], [0, 1]])
MEAN2 = np.array([1, -2])
COV2 = np.array([[1, 0], [0, 1]])

# Generate random points from the two components
X1 = multivariate_normal.rvs(MEAN1, COV1, CLASS1_SIZE)
X2 = multivariate_normal.rvs(MEAN2, COV2, CLASS2_SIZE)

# Combine the points and generate labels
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))

# Plot the generated data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Generated Data')
plt.show()

# Split data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Model parameters
n_features = X_train.shape[1]
n_output = 1

# Initialize weights and biases
W0 = np.array(np.zeros(1))
W1 = np.array([np.random.randn(1) * 0.1, np.random.randn(1) * 0.1]).reshape(-1, 2)


# Create nodes
x1_node = Input()
y_node = Input()

w0_node = Parameter(W0)
w1_node = Parameter(W1)

# Build computation graph
b_node = Linear(w1_node, x1_node, w0_node)
sigmoid = Sigmoid(b_node)
loss = BCE(y_node, sigmoid)

# Create graph outside the training loop
graph = [x1_node, w0_node, w1_node, b_node, sigmoid, loss]
trainable = [w0_node, w1_node]


# Forward and Backward Pass
def forward_pass(graph):
    for n in graph:
        n.forward()


def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()


# SGD Update
def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value -= np.dot(learning_rate, t.gradients[t].reshape(1, -1))[0]

# Dictionary to store loss values for different batch sizes
loss_values = {batch_size: [] for batch_size in [1,2,4,8,16,32,64,128]}

# Training loop for different batch sizes
for batchs in [1,2,4,8,16,32,64,128]:
    for epoch in range(EPOCHS):
        loss_value = 0
        for i in range(0, X_train.shape[0], batchs):
            end = min(batchs + i, X_train.shape[0])
            x1_node.value = X_train[i:end].T
            y_node.value = y_train[i:end].reshape(1, -1)

            forward_pass(graph)
            backward_pass(graph)
            sgd_update(trainable, LEARNING_RATE)

            loss_value += loss.value
        loss_values[batchs].append(loss_value / X_train.shape[0])
        print(f"Epoch {epoch + 1}, Loss: {loss_value / X_train.shape[0]}")

    # Evaluate the model
    correct_predictions = 0
    for i in range(X.shape[0]):
        x1_node.value = X[i].reshape(2, -1)
        forward_pass(graph)

        pre = 1 if sigmoid.value[0][0] >= 0.5 else 0
        if pre == y[i]:
            correct_predictions += 1

    accuracy = correct_predictions / X.shape[0]
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Plot decision boundary
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
    Z = []
    for i, j in zip(xx.ravel(), yy.ravel()):
        x1_node.value = np.array([i, j]).reshape(2, -1)
        forward_pass(graph)
        Z.append(sigmoid.value)
    Z = np.array(Z).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

    # Reset weights for next batch size
    w0_node.value = np.zeros((1))
    w1_node.value = np.array([np.random.randn(1) * 0.1, np.random.randn(1) * 0.1]).reshape(-1, 2)

# Plot loss curves for different batch sizes
plt.figure(figsize=(10, 6))
for batch_size, losses in loss_values.items():
    plt.plot(range(EPOCHS), losses, label=f'Batch size {batch_size}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve for Different Batch Sizes')
plt.legend()
plt.grid()
plt.show()
