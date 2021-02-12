"""Image classifier - fashionmnist dataset from Tensorflow"""

"""https://www.tensorflow.org/datasets/catalog/fashion_mnist"""

import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import layers

if __name__ == '__main__':
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    print(len(train_images), len(train_labels))
    print(train_images.shape)
    # print(train_labels[0])
    # plt.imshow(train_images[0], cmap='gray', vmin=0, vmax=255)
    # plt.show()

    model = keras.Sequential()

    """ test acc: 78% """
    model.add(layers.Conv2D(5, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPool2D(pool_size=2, strides=2))
    model.add(layers.Conv2D(8, kernel_size=5, strides=1, activation='relu'))
    model.add(layers.MaxPool2D(pool_size=2, strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(72, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    """--------"""

    # model.add(layers.Flatten(input_shape=(28, 28)))

    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    train_images = train_images.reshape(60000, 28, 28, 1)
    model.fit(train_images, train_labels, epochs=20)

    test_images = test_images.reshape(len(test_images), 28, 28, 1)
    score = model.evaluate(test_images, test_labels, verbose=2)
    print(score)
# model.summary()
