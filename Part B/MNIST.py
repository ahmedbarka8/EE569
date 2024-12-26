from matplotlib import pyplot as plt

from EDF import *
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import time

# Define constants hyperparameters
LEARNING_RATE = 0.01
EPOCHS = 100
TEST_SIZE = 0.4
Batch_Size = 64

# Load the data base
mnist = datasets.load_digits()
X, y_1 = mnist['data'], mnist['target'].astype(int)

X = X / 16
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y_1.reshape(-1, 1))

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
hidden_size = 64
output_size = 10

# Create nodes
x_node = Input()
y_node = Input()

layer_1 = Linear(x_node, hidden_size, n_features)
layer1 = Sigmoid(layer_1)
layer_2 = Linear(layer1, output_size, hidden_size)
output = SoftMax(layer_2)
loss = CE(y_node, output)


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
        t.value -= learning_rate * t.gradients[t]


# create the graph and list for the trainable nodes
graph = topological_sort(loss)
trainable = [i for i in graph if isinstance(i, Parameter)]

tick = time.time()
loss_values = []

# Training
for epoch in range(EPOCHS):
    loss_value = 0
    for i in range(0, X_train.shape[0], Batch_Size):
        end = min(Batch_Size + i, X_train.shape[0])
        x_node.value = X_train[i:end].T
        y_node.value = y_train[i:end].T

        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, LEARNING_RATE)

        loss_value += loss.value
    loss_values.append(loss_value / X_train.shape[0])
    print(f"Epoch {epoch + 1}, Loss: {loss_value / X_train.shape[0]}")

LEARNING_TIME = time.time() - tick
tick = time.time()

# Evaluate the model
correct_predictions = 0
for i in range(X.shape[0]):
    x_node.value = X[i].reshape(-1, 1)
    forward_pass(graph)

    indx_pre = np.argmax(output.value)
    indx_true = np.argmax(y[i])
    if indx_pre == indx_true:
        correct_predictions += 1

accuracy = correct_predictions / X.shape[0]
Testing_Time = time.time() - tick

print()
print(f"Epochs: {EPOCHS} , Learning_Rate: {LEARNING_RATE} , Batch_Size : {Batch_Size}")
print(f"Learning_Time : {LEARNING_TIME:.2f} Second, Testing_Time : {Testing_Time:.2f} Second")
print(f"Average_Loss : {loss_value / (X_train.shape[0])} , Accuracy: {accuracy * 100:.2f}%")

plt.plot(range(EPOCHS), loss_values)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.grid()
plt.show()
