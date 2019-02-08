from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.keras import layers

print(tf.VERSION)
print(tf.keras.__version__)

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)

print(len(train_labels))

print(train_labels)

print(test_images.shape)


print(len(test_labels))

plt.figure()
plt.imshow(train_images[50], dtype='unit64')
plt.colorbar()
plt.grid(False)
plt.show()

# data = [[0, 0.25], [0.5, 0.75]]

# fig, ax = plt.subplots()
# im = ax.imshow(data, cmap=plt.get_cmap('hot'), interpolation='nearest',
#                vmin=0, vmax=1)
# fig.colorbar(im)
# plt.show()
