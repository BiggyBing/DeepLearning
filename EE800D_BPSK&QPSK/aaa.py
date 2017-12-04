#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
import BPSK_QPSK_Generator_New
import numpy as np



acc=[]
loss=[]
for k in range(2,13):
    B_S = 12 
    B_forUse = k
    BPSK,QPSK,EPSK = BPSK_QPSK_Generator_New.BPSK_QPSK_Generator(Bit_size = B_S, Num_shift = 10, Sample_Noise = True, Disturbance = True, Shift = True)
    
    
    
    Data = []
    reshapeType = 'Random'
    for i in range(BPSK.shape[0]):
        temp = data_reshape(BPSK[i,0:B_forUse*4],B_forUse,reshapeType = reshapeType)
        Data.append(temp)
        #Data = np.dstack((Data,temp))    
    
    for i in range(QPSK.shape[0]):
        temp = data_reshape(QPSK[i,0:B_forUse*4],B_forUse,reshapeType = reshapeType)
        Data.append(temp)
        
    for i in range(EPSK.shape[0]):
        temp = data_reshape(EPSK[i,0:B_forUse*4],B_forUse,reshapeType = reshapeType)
        Data.append(temp)
        
    if reshapeType == 'Random':
        data = np.array(Data).reshape((len(Data),4,-1,1))  
    else:
        data = np.array(Data).reshape((len(Data),2,-1,1))
    
    BPSK_label = BPSK[:,B_S*4:]
    QPSK_label = QPSK[:,B_S*4:]
    EPSK_label = EPSK[:,B_S*4:]
    
    label = np.concatenate((BPSK_label,QPSK_label,EPSK_label))
    
    model = Sequential()
        
    model.add(Conv2D(32, (2,2), padding='same', activation='relu', input_shape = data.shape[1:]))
    model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    
    model.add(Conv2D(64, (1,1), padding='same', activation='relu'))
    model.add(Conv2D(64, (1,1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(data, label,epochs=10,batch_size=64,shuffle=True)
    acc.append(hist.history['acc'][9])
    loss.append(hist.history['loss'][9])