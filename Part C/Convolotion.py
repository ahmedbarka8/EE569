from EDF import *
import numpy as np
from keras.src.datasets import mnist
from matplotlib import pyplot


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

x_node = Input()
y_node = Input()

conv = Conv(x_node,1,1)
maxp = MaxPooling(conv)


conv.kernels.value = np.array([[[1,0,-1,
                           2,0,-2,
                           1,0,-1]]]).reshape(3,3,1,1)


arr = []
x_node.value = train_X[0:9].reshape(9, 28, 28, 1)
conv.forward()
arr.append(conv.value.reshape(9,28,28))

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(arr[0][i], cmap=pyplot.get_cmap('gray'))
pyplot.show()

conv.kernels.value = np.array([[[0,1,0,
                  1,4,1,
                  0,1,0]]]).reshape(3,3,1,1)

for i in range(9,18):
    pyplot.subplot(330 + 1 + i- 9)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()

arr = []
x_node.value = train_X[9:18].reshape(9, 28, 28, 1)
conv.forward()
arr.append(conv.value.reshape(9,28,28))

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(arr[0][i], cmap=pyplot.get_cmap('gray'))
pyplot.show()