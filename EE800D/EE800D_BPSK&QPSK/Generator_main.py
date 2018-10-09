#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 12:53:44 2018

@author: bingyangwen
"""

from Generator_function import prediction, data_generator


train_feature, train_label, test_feature, test_label = data_generator(12, sample = True)

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization


number_of_classes = 3
y_train = np_utils.to_categorical(train_label, number_of_classes)
y_test = np_utils.to_categorical(test_label_1, number_of_classes)


#%%
model = Sequential()

model.add(Conv1D(32, 2, input_shape = (12,1)))
model.add(Activation('relu'))

BatchNormalization(axis = -1)

model.add(Conv1D(32, 2))
model.add(Activation('relu'))

BatchNormalization(axis = -1)

model.add(Conv1D(64, 2))
model.add(Activation('relu'))

model.add(Flatten())
BatchNormalization(axis = -1)


model.add(Dense(1024, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(train_feature_1, y_train, batch_size=128, nb_epoch=10)


#%%

error_1, prediction_1 = prediction(model, test_feature_1, y_test)
#%%
from sklearn.metrics import confusion_matrix
y_true = []
y_pred = []
for item in prediction_1:
    y_true.append(item[1])
    y_pred.append(item[2])
confusion_matrix(y_true, y_pred, labels = ["BPSK","QPSK","16PSK"])
    