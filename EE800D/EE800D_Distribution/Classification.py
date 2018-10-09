import Norm_Uni_Ex

import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model

train, test = Norm_Uni_Ex.dataset()
#a=Norm_Uni_Ex.dataset()
#
#X_train = train[:,0:784]
#y_train = train[:,785]
#X_test = test[:,0:784]
#y_test = test[:,785]