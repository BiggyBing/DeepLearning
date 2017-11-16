#Classification by TFLearn
import tflearn
import os
import Norm_Uni_Ex
import numpy as np

def onehot(input):
    x = np.zeros((len(input),max(input)+1)).astype('int')

    
    for i in range(len(input)):
        
        for j in range(max(input)+1):
            
            if input[i] == j:
                x[i,j]=1
    return x
        
        

def vgg16(input, num_class):

    #in the model, we added trainable=False to make sure the parameter are not updated during training
    x = tflearn.conv_2d(input, 32, 3, activation='relu', scope='conv1_1')
    x = tflearn.conv_2d(x, 32, 3, activation='relu', scope='conv1_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv2_1')
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv2_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = tflearn.fully_connected(x, 512, activation='relu', scope='fc1')
    x = tflearn.dropout(x, 0.5, name='dropout1')
    #we changed the structure here to let the fc only have 2048, less parameter, enough for our task
    x = tflearn.fully_connected(x, num_class, activation='softmax', scope='fc2')

    return x





train, test = Norm_Uni_Ex.dataset(datasize = 784)
X_train = train[:,0:784]
y_train = train[:,784].astype('int')
y_train = onehot(y_train)

X_test = test[:,0:784]
y_test = test[:,784].astype('int')
y_test = onehot(y_test)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

num_classes = 3

x = tflearn.input_data(shape=[None, 28, 28, 1], name='input')

softmax = vgg16(x, num_classes)
regression = tflearn.regression(softmax, optimizer='adam',
                                loss='categorical_crossentropy',
                                learning_rate=0.0001)

model = tflearn.DNN(regression, tensorboard_verbose=0)


# Start finetuning
model.fit(X_train, y_train, n_epoch=20, validation_set=(X_test, y_test), shuffle=True,
          show_metric=True, batch_size=64, run_id='Distribution_3')


##let's just test if the model can predict image right
# model.predict(img_array) to see the final result
