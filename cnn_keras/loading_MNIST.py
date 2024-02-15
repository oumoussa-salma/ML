import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train/255
x_test = x_test/255

print(x_train.shape)
print(y_train.shape)

plt.imshow(x_train[600], cmap='Greys')
plt.show()
plt.imshow(x_train[1000], cmap='Greys')
plt.show()
plt.imshow(x_train[0], cmap='Greys')
plt.show()

