import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.utils import np_utils
from keras.datasets import mnist
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
nb_classes = 10

## Importing the MNIST dataset using Keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()

## Reshaping inputs (each image is 28 x 28, and there are 60000 training images
## and 10000 test images.
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

## Data normalization below

X_train = X_train.astype('float')
X_test = X_test.astype('float')
X_train /= 255
X_test /= 255

## Transform labels into a one hot vector

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(100, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))

## Last input layer
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
              metrics=['accuracy'])


fit = model.fit(X_train, y_train,
                    batch_size=128, nb_epoch=11,
                    verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])