import numpy as np


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
        self.gradients[A] = np.matmul(self.outputs[0].gradients[self], x.value.T).reshape(self.W.value.shape)
        self.gradients[b] = np.sum(self.outputs[0].gradients[self], axis=1, keepdims=True).reshape(self.b.value.shape)


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


class BCE(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        y_pred_clipped = np.clip(y_pred.value, 1e-15, 1 - 1e-15)
        self.value = np.sum(-y_true.value * np.log(y_pred_clipped) - (1 - y_true.value) * np.log(1 - y_pred_clipped))

    def backward(self):
        y_true, y_pred = self.inputs
        y_pred_clipped = np.clip(y_pred.value, 1e-15, 1 - 1e-15)
        self.gradients[y_pred] = (1 / y_true.value.shape[1]) * (y_pred_clipped - y_true.value) / (
                    y_pred.value * (1 - y_pred_clipped))
        self.gradients[y_true] = (1 / y_true.value.shape[1]) * (np.log(y_pred.value) - np.log(1 - y_pred.value))


class SoftMax(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def forward(self):
        input_value = self.inputs[0].value
        exp_values = np.exp(input_value - np.max(input_value, axis=0, keepdims=True))
        self.value = exp_values / np.sum(exp_values, axis=0, keepdims=True)

    def backward(self):
        self.gradients[self.inputs[0]] = (self.value - self.outputs[0].gradients[self])


class BCE_Soft(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        y_pred_clipped = np.clip(y_pred.value, 1e-15, 1 - 1e-15)
        self.value = np.sum(-y_true.value * np.log(y_pred_clipped)) / y_true.value.shape[0]

    def backward(self):
        y_true, y_pred = self.inputs
        y_pred_clipped = np.clip(y_pred.value, 1e-15, 1 - 1e-15)
        self.gradients[y_pred] = y_true.value
        self.gradients[y_true] = 0
