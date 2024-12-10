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

#Creating a Linear node
class Linear(Node):
    def __init__(self,A,x,b):
        # Initialize with three inputs A and x and b
        Node.__init__(self,[A,x,b])

    def forward(self):
        # Perform element-wise addition and multiplication
        A,x,b = self.inputs
        self.value = np.matmul(A.value , x.value) + b.value.reshape(-1,1)

    def backward(self):
        # Compute gradients for A,x and b based on the chain rule
        A, x, b = self.inputs
        self.gradients[x] = np.matmul(A.value.T,self.outputs[0].gradients[self])
        self.gradients[A] = np.matmul(self.outputs[0].gradients[self],x.value.T)
        self.gradients[b] = np.sum(self.outputs[0].gradients[self], axis=1)


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

class Multiply(Node):
    def __init__(self, x, y):
        # Initialize with two inputs x and y
        Node.__init__(self, [x, y])

    def forward(self):
        # Perform element-wise multiplication
        x, y = self.inputs
        self.value = x.value * y.value

    def backward(self):
        # Compute gradients for x and y based on the chain rule
        x, y = self.inputs
        self.gradients[x] = self.outputs[0].gradients[self] * y.value
        self.gradients[y] = self.outputs[0].gradients[self] * x.value

class Addition(Node):
    def __init__(self, x, y):
        # Initialize with two inputs x and y
        Node.__init__(self, [x, y])

    def forward(self):
        # Perform element-wise addition
        x, y = self.inputs
        self.value = x.value + y.value

    def backward(self):
        # The gradient of addition with respect to both inputs is the gradient of the output
        x, y = self.inputs
        self.gradients[x] = self.outputs[0].gradients[self]
        self.gradients[y] = self.outputs[0].gradients[self]


# Sigmoid Activation Node
class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x,-500,500)))

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
        y_pred_clipped = np.clip(y_pred.value, 1e-15, 1 - 1e-15 )
        self.value = np.sum(-y_true.value*np.log(y_pred_clipped)-(1-y_true.value)*np.log(1-y_pred_clipped))

    def backward(self):
        y_true, y_pred = self.inputs
        self.gradients[y_pred] = (1 / y_true.value.shape[1]) * (y_pred.value - y_true.value)/(y_pred.value*(1-y_pred.value))
        self.gradients[y_true] = (1 / y_true.value.shape[1]) * (np.log(y_pred.value) - np.log(1-y_pred.value))
