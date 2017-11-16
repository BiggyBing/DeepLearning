#Vgg by keras, a simulated code by FVGG_Emo.py

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
import Norm_Uni_Ex


train, test = Norm_Uni_Ex.dataset(250000)
X_train = train[:,0:250000]
y_train = train[:,250000]
X_test = test[:,0:250000]
y_test = test[:,250000]
#X_train = []
#y_train = []
#X_test = []
#y_test = []
#for i in range(len(train)):
#    X_train.append(train[i][0:784])
#    y_train.append(train[i][784])
#for j in range(len(test)):
#    X_test.append(train[j][0:784])
#    y_test.append(train[j][784])


X_train = X_train.reshape(X_train.shape[0], 500, 500, 1)
X_test = X_test.reshape(X_test.shape[0], 500, 500, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

number_of_classes = 3
#Converts a class integer to binary matrix(one-hot)
Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)

model = Sequential()

#Layer 1 
#After first layer, you don't need to specify the size of the input anymore
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Layer 3
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Layer 4
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Fully connected layer
model.add(Flatten())

# dense implement: output = activation(dot(input, kernel) + bias) 
# 512 perceptrons, fully connected layer
model.add(Dense(512, activation='relu'))
#BatchNormalization()
model.add(Dropout(0.5))
model.add(Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=128, nb_epoch=10, validation_data=(X_test, Y_test))