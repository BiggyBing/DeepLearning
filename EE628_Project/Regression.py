import CNN_Regression
import pandas as pd
import numpy as np
from keras.optimizers import Adam

Dir = '/Users/bingyangwen/Desktop/deep learning/'
train=pd.read_json(Dir + 'train.json')
test = pd.read_json(Dir + 'test.json')


X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75, 1) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75, 1) for band in train["band_2"]])


X_angle_mean = np.array([])
for inc_angle in train["inc_angle"]:
    if inc_angle == 'na':
        continue
    X_angle_mean = np.append(X_angle_mean, float(inc_angle))
mean = np.mean(X_angle_mean)
  
X_angle = np.array([])
for inc_angle in train["inc_angle"]:
    if inc_angle == 'na':
        inc_angle = mean
    X_angle = np.append(X_angle, float(inc_angle))
    
        



model = CNN_Regression.CNN_Regression((75,75,1))
model.compile(loss='mean_squared_error', optimizer=Adam())
model.fit(X_band_1, X_angle, batch_size=128, nb_epoch=50)

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
