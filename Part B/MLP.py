from EDF import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define constants hyperparameters
CLASS1_SIZE = 100
CLASS2_SIZE = 100
LEARNING_RATE = 0.1
EPOCHS = 100
TEST_SIZE = 0.25

# Define the means and covariances of the two components
MEAN1 = np.array([-5, -5])
COV1 = np.array([[1, 0], [0, 1]])
MEAN2 = np.array([5, 5])
COV2 = np.array([[1, 0], [0, 1]])
MEAN3 = np.array([5, -5])
COV3 = np.array([[1, 0], [0, 1]])
MEAN4 = np.array([-5, 5])
COV4 = np.array([[1, 0], [0, 1]])

# Generate random points from the two components
X1 = multivariate_normal.rvs(MEAN1, COV1, CLASS1_SIZE // 2)
X2 = multivariate_normal.rvs(MEAN2, COV2, CLASS2_SIZE // 2)
X3 = multivariate_normal.rvs(MEAN3, COV3, CLASS1_SIZE // 2)
X4 = multivariate_normal.rvs(MEAN4, COV4, CLASS2_SIZE // 2)

# Combine the points and generate labels
X = np.vstack((X1, X2, X3, X4))
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
batches = 1

# Create nodes
x1_node = Input()
y_node = Input()

layer_1 = Linear(x1_node, 20, 2)
layer1 = Sigmoid(layer_1)
layer_2 = Linear(layer1, 20, 20)
layer2 = Sigmoid(layer_2)
layer_3 = Linear(layer2, 1, 20)
output = Sigmoid(layer_3)
loss = BCE(y_node, output)


# create a graph Automatically
def topological_sort(node, visited=None, graph=None):
    if not visited:
        visited = set()
        graph = []
    if node in visited:
        return
    visited.add(node)
    for input_node in node.inputs:
        graph = topological_sort(input_node, visited, graph)
    graph.append(node)
    return graph


# Forward and Backward Pass
def forward_pass(graph):
    for n in graph:
        n.forward()


def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()


# SGD Update
def sgd_update(trainable, learning_rate=1e-2):
    for t in trainable:
        t.value -= np.dot(learning_rate, t.gradients[t])


# create the graph and list for the trainable nodes
graph = topological_sort(loss)
trainable = [i for i in graph if isinstance(i, Parameter)]

# Dictionary to store loss values for different batch sizes


# Training loop for different batch sizes
for epoch in range(EPOCHS):
    loss_value = 0
    for i in range(0, X_train.shape[0], batches):
        end = min(batches + i, X_train.shape[0])
        x1_node.value = X_train[i:end].T
        y_node.value = y_train[i:end].reshape(1, -1)

        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, LEARNING_RATE)

        loss_value += loss.value
    print(f"Epoch {epoch + 1}, Loss: {loss_value / X_train.shape[0]}")

# Evaluate the model
correct_predictions = 0
for i in range(X.shape[0]):
    x1_node.value = X[i].reshape(2, -1)
    forward_pass(graph)

    pre = 1 if output.value[0][0] >= 0.5 else 0
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
    Z.append(output.value)
Z = np.array(Z).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()
