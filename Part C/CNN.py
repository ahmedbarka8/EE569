from matplotlib import pyplot as plt

from EDF import *
import numpy as np
import time
from keras.src.datasets import mnist
from sklearn.preprocessing import OneHotEncoder

LEARNING_RATE = 0.1
EPOCHS = 3

# Load the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Preprocess the data
train_X = train_X / 255.0
test_X = test_X / 255.0

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False)
train_y = encoder.fit_transform(train_y.reshape(-1, 1))
test_y = encoder.fit_transform(test_y.reshape(-1, 1))

# Model parameters
batches = 64

# Define CNN architecture
x_node = Input()
y_node = Input()

conv1 = Conv(x_node,1, 16)
relu1 = ReLU(conv1)
pool1 = MaxPooling(relu1)

conv2 = Conv(pool1,16, 32)
relu2 = ReLU(conv2)
pool2 = MaxPooling(relu2)

conv3 = Conv(pool2,32, 64)
relu3 = ReLU(conv3)
pool3 = MaxPooling(relu3)

conv4 = Conv(pool3,64,128)
relu4 = ReLU(conv4)

flat = Flatten(relu4)
output_layer = Linear(flat, 10, 1152)
output = SoftMax(output_layer)

loss = CE(y_node, output)

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

# Training loop
graph = topological_sort(loss)
trainable = [n for n in graph if isinstance(n, Parameter)]
loss_values = []

print("Learning ....")
tick = time.time()
for epoch in range(EPOCHS):
    loss_value = 0
    for i in range(0, train_X.shape[0], batches):
        end = min(batches + i, train_X.shape[0])
        if end == train_X.shape[0]:
            break
        x_node.value = train_X[i:end].reshape(-1,28,28,1)
        y_node.value = train_y[i:end].T

        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, LEARNING_RATE)


        loss_value += loss.value
    loss_values.append(loss_value / test_X.shape[0])
    print(f"Epoch {epoch + 1}, Loss: {loss_value / (train_X.shape[0])}")

LEARNING_TIME = time.time() - tick
tick = time.time()

print("Testing ...")
# Evaluate the model
correct_predictions = 0
for i in range(test_X.shape[0]):
    x_node.value = test_X[i:i+1].reshape(1,28,28,1)
    forward_pass(graph)

    indx_pre = np.argmax(output.value)
    indx_true = np.argmax(test_y[i])
    if indx_pre == indx_true:
        correct_predictions += 1

accuracy = correct_predictions / test_X.shape[0]
Testing_Time = time.time() - tick

print()
print(f"Epochs: {EPOCHS} , Learning_Rate: {LEARNING_RATE} , Batch_Size : {batches}")
print(f"Learning_Time : {LEARNING_TIME/60:.2f} Minute, Testing_Time : {Testing_Time/60:.2f} Minute")
print(f"Average_Loss : {loss_value / (train_X.shape[0])} , Accuracy: {accuracy * 100:.2f}%")

plt.plot(range(EPOCHS), loss_values)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.grid()
plt.show()