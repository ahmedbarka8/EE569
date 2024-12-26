import numpy as np
from typing import Tuple


# Base Node class
class Node:
    def __init__(self, inputs=None):
        if inputs is None:
            inputs = []
        self.inputs = inputs
        self.outputs = []
        self.value = None
        self.gradients = {}

        for node in inputs:
            node.outputs.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


#Linear node
class Linear(Node):
    def __init__(self, x, width, input_size):
        # Initialize with three inputs A and x and b
        factor = 1 / np.sqrt(input_size)
        self.W = Parameter(np.random.randn(width, input_size) * factor)
        self.b = Parameter(np.zeros((width, 1)))
        Node.__init__(self, [self.W, self.b, x])

    def forward(self):
        # Perform element-wise addition and multiplication
        A, b, x = self.inputs
        self.value = np.matmul(A.value, x.value) + b.value.reshape(-1, 1)

    def backward(self):
        # Compute gradients for A,x and b based on the chain rule
        A, b, x = self.inputs
        self.gradients[x] = np.matmul(A.value.T, self.outputs[0].gradients[self])
        self.gradients[A] = np.matmul(self.outputs[0].gradients[self], x.value.T)
        self.gradients[b] = np.sum(self.outputs[0].gradients[self], axis=1, keepdims=True)


# Input Node
class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]


# Parameter Node
class Parameter(Node):
    def __init__(self, value):
        Node.__init__(self)
        self.value = value

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]


# Sigmoid Activation Node
class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self):
        input_value = self.inputs[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        partial = self.value * (1 - self.value)
        self.gradients[self.inputs[0]] = partial * self.outputs[0].gradients[self]



class SoftMax(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def forward(self):
        input_value = self.inputs[0].value
        exp_values = np.exp(input_value - np.max(input_value, axis=0, keepdims=True))
        self.value = exp_values / np.sum(exp_values, axis=0, keepdims=True)

    def backward(self):
        self.gradients[self.inputs[0]] = self.value * (self.outputs[0].gradients[self]
                                                       - np.sum(self.value * self.outputs[0].gradients[self], axis=0,
                                                                keepdims=True))


class CE(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        y_pred_clipped = np.clip(y_pred.value, 1e-15, 1 - 1e-15)
        self.value = np.sum(-y_true.value * np.log(y_pred_clipped)) / y_true.value.shape[1]

    def backward(self):
        y_true, y_pred = self.inputs
        y_pred_clipped = np.clip(y_pred.value, 1e-15, 1 - 1e-15)
        self.gradients[y_pred] = (y_pred.value - y_true.value) / y_true.value.shape[1]
        self.gradients[y_true] = 0


class Conv(Node):
    def __init__(self, x,num, num_filters):


        self.num_filters = num_filters
        self.factor = (1 / np.sqrt(3 * 3 * num))  # He initialization
        self.kernels = Parameter(np.random.randn(3, 3, num, num_filters) * self.factor)
        self.b = Parameter(np.zeros(num_filters))
        Node.__init__(self,[x,self.kernels,self.b])

    def forward(self):
        x, kernel, b = self.inputs
        n, h_in, w_in ,_= x.value.shape
        h_f, w_f, _, n_f = kernel.value.shape
        self.value = np.zeros((n,h_in,w_in,n_f))
        padded_x = np.pad(x.value,((0, 0), (1,1), (1,1),(0,0)),mode='constant')

        for i in range(h_in):
            for j in range(w_in):

                self.value[:,i,j,:] = np.tensordot(padded_x[:,i:i+3,j:j+3,:],self.kernels.value,
                                                   axes=((1,2,3),(0,1,2)))


        self.value += self.b.value.reshape((1,1,1,n_f))

    def backward(self):
        x, kernels, biases = self.inputs
        gradient = self.outputs[0].gradients[self]
        _, h_out, w_out, _ = gradient.shape
        n, h_in, w_in, _ = x.value.shape
        h_f, w_f, _, _ = self.kernels.value.shape
        padded_x = np.pad(x.value, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant')

        self.gradients[x] = np.zeros_like(padded_x)
        self.gradients[kernels] = np.zeros_like(kernels.value)
        self.gradients[biases] = np.zeros_like(biases.value)

        self.gradients[biases] += self.outputs[0].gradients[self].sum(axis=(0, 1, 2))

        for i in range(h_out):
            for j in range(w_out):

                self.gradients[x][:, i:i+3, j:j+3, :] += np.tensordot(
                    gradient[:, i, j, :],
                    self.kernels.value,
                    axes=((1),(3))
                )
                self.gradients[kernels] += np.tensordot(
                    padded_x[:, i:i+3, j:j+3, ],
                    gradient[:, i, j,:],
                    axes=((0),(0))
                )

        self.gradients[x] = self.gradients[x][:,1:-1,1:-1,:]

class MaxPooling(Node):
    def __init__(self, x):
        self.cache = {}
        Node.__init__(self, [x])

    def forward(self):
        x = self.inputs[0].value
        n, h_in, w_in, c = x.shape
        h_out = 1 + (h_in - 2) // 2
        w_out = 1 + (w_in - 2) // 2
        self.value = np.zeros((n, h_out, w_out, c))

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * 2
                h_end = h_start + 2
                w_start = j * 2
                w_end = w_start + 2
                a_prev_slice = x[:, h_start:h_end, w_start:w_end, :]
                self._save_mask(x=a_prev_slice, cords=(i, j))
                self.value[:, i, j, :] = np.max(a_prev_slice, axis=(1, 2))

    def backward(self):
        x = self.inputs[0]
        gradient = self.outputs[0].gradients[self]
        self.gradients[x] = np.zeros_like(x.value)
        _, h_out, w_out, _ = gradient.shape

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * 2
                h_end = h_start + 2
                w_start = j * 2
                w_end = w_start + 2
                self.gradients[x][:, h_start:h_end, w_start:w_end, :] += \
                    gradient[:, i:i + 1, j:j + 1, :] * self.cache[(i, j)]

    def _save_mask(self, x: np.array, cords: Tuple[int, int]) -> None:
        mask = np.zeros_like(x)
        n, h, w, c = x.shape
        x = x.reshape(n, h * w, c)
        idx = np.argmax(x, axis=1)

        n_idx, c_idx = np.indices((n, c))
        mask.reshape(n, h * w, c)[n_idx, idx, c_idx] = 1
        self.cache[cords] = mask


class Flatten(Node):
    def __init__(self, x):
        Node.__init__(self, [x])

    def forward(self):
        x = self.inputs[0].value
        self.value = np.ravel(x).reshape(x.shape[0], -1).T

    def backward(self):
        x = self.inputs[0]
        self.gradients[x] = self.outputs[0].gradients[self].T.reshape(x.value.shape)


class ReLU(Node):
    def __init__(self, x):
        Node.__init__(self, [x])

    def forward(self):
        x = self.inputs[0].value
        self.value = np.maximum(0, x)

    def backward(self):
        x = self.inputs[0].value
        self.gradients[self.inputs[0]] = (x > 0) * self.outputs[0].gradients[self]
