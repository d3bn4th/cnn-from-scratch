import numpy as np
from keras.datasets import mnist
from conv import Conv3x3
from maxpool import MaxPool2

# Load the MNIST dataset using Keras
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

conv = Conv3x3(8)
pool = MaxPool2()

output = conv.forward(train_images[0])
print(output.shape) # (26, 26, 8)

output = pool.forward(output)
print(output.shape) # (13, 13, 8)