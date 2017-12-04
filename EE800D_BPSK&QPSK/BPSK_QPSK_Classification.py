from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
import BPSK_QPSK_Generator_New
import numpy as np



B_S = 12 
B_forUse = 4
#BPSK,QPSK,EPSK = BPSK_QPSK_Generator_New.BPSK_QPSK_Generator(Bit_size = B_S, Num_shift = 10, Sample_Noise = True, Disturbance = True, Shift = True)

def data_reshape(data, bit_size):
    i = 0
    upper = np.array([])
    lower = np.array([])
    
    while i < bit_size * 4 - 1:
        upper = np.append(upper, data[i])
        lower = np.append(lower, data[i+1])
        i=i+2
    new = np.append(upper, lower)
    new = new.reshape(2,-1)
    return new.tolist()

Data = []

for i in range(BPSK.shape[0]):
    temp = data_reshape(BPSK[i,0:B_forUse*4],B_forUse)
    Data.append(temp)
    #Data = np.dstack((Data,temp))    

for i in range(QPSK.shape[0]):
    temp = data_reshape(QPSK[i,0:B_forUse*4],B_forUse)
    Data.append(temp)
    
for i in range(EPSK.shape[0]):
    temp = data_reshape(EPSK[i,0:B_forUse*4],B_forUse)
    Data.append(temp)

data = np.array(Data).reshape((len(Data),2,B_forUse*2,1))   

BPSK_label = BPSK[:,B_S*4:]
QPSK_label = QPSK[:,B_S*4:]
EPSK_label = EPSK[:,B_S*4:]

label = np.concatenate((BPSK_label,QPSK_label,EPSK_label))

model = Sequential()
    
model.add(Conv2D(32, (2,2), padding='same', activation='relu', input_shape = (2,B_forUse*2,1)))
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
model.fit(data, label,epochs=20,batch_size=64,shuffle=True)

acc =[0.50042317708333328,
 0.78419596354166665,
 0.85016276041666672,
 0.89995930989583328,
 0.92674153645833335,
 0.94802246093749998,
 0.9530029296875,
 0.978515625,
 0.9838541666666667,
 0.98991699218749996,
 0.99205729166666667,]

loss =[0.80543660915767157,
 0.48532157350952426,
 0.26485995209465424,
 0.17715780224728708,
 0.13471842824462024,
 0.090900648148544858,
 0.073207950354412549,
 0.045078086411007114,
 0.026921813547891613,
 0.019064385595512098,
 0.013844236250333642]

