from EDF import *
import numpy as np
from keras.src.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot

LEARNING_RATE = 0.1
EPOCHS = 20

#loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

#printing the shapes of the vectors
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()

encoder = OneHotEncoder(sparse_output=False)
train_y = encoder.fit_transform(train_y.reshape(-1, 1))
test_y = encoder.fit_transform(test_y.reshape(-1, 1))

# Flatten and normalize
train_X = train_X.reshape(train_X.shape[0], 784) / 255.0
test_X = test_X.reshape(test_X.shape[0], 784) / 255.0

# Model parameters
n_features = train_X.shape[1]
hidden_size1 = 128
hidden_size2 = 64
output_size = 10
batches = 64

# Create nodes
x_node = Input()
y_node = Input()



hidden1 = Linear(x_node, hidden_size1, n_features)
act1 = Sigmoid(hidden1)

hidden2 = Linear(act1, hidden_size2, hidden_size1)
act2 = Sigmoid(hidden2)

output_layer = Linear(act2, output_size, hidden_size2)
output = SoftMax(output_layer)

loss = BCE_Soft(y_node, output)


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

# Training loop for different batch sizes
for epoch in range(EPOCHS):
    loss_value = 0
    for i in range(0, train_X.shape[0], batches):
        end = min(batches + i, train_X.shape[0])
        x_node.value = train_X[i:end].T
        y_node.value = train_y[i:end].T

        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, LEARNING_RATE)

        loss_value += loss.value
    print(f"Epoch {epoch + 1}, Loss: {loss_value / test_X.shape[0]}")

# Evaluate the model
correct_predictions = 0
for i in range(test_X.shape[0]):
    x_node.value = test_X[i].T
    forward_pass(graph)

    indx_pre = np.argmax(output.value)
    indx_true = np.argmax(test_y[i])
    if indx_pre == indx_true:
        correct_predictions += 1

accuracy = correct_predictions / test_X.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")
