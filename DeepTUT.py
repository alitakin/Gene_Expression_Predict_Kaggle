# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 15:11:33 2017

@author: pia
"""


from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dropout, Flatten, Dense,Activation
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np
# for temporary use only
from random import randint


def get_data(amount,i,f,f2):
    x_train = []
    y_train = []
        
#    for line in f:
#        if (i>0):
#            line = line.split('\n')[0]
#            row = line.split(",")[1:]
##            row = row.split('\n')[0]
#            print(row)
#        i=i+1
#        if (i>1000): break
    # filter size of convolution2D
#    w, h = amount, amount
    w, h = 100,5
    x_train=np.random.random((amount, w, h, 32))
    x_train=np.array(x_train)
    print(x_train.shape)

#    for line in f2:
#        line = line.split('\n')[0]
#        prob = line.split(",")[1]
#        print(prob)
#        y_train.append([prob])
#        if (i%amount==0): break
#        i=i+1


    for i in range(0,amount):
        y_train.append([randint(0,1)])
#        y_train.append([1])
    y_train = np.array(y_train)
    print(y_train.shape)
    y_train = np_utils.to_categorical(y_train, 2)
    return x_train, y_train



def get_data2(amount,i,f):
    x_train = []
    y_train = []
        

    # filter size of convolution2D
    w, h = amount, amount
    x_train=np.random.random((amount, w, h, 32))
    x_train=np.array(x_train)

    for i in range(0,amount):
        y_train.append([randint(0,1)])
#        y_train.append([1])
    y_train = np.array(y_train)
    y_train = np_utils.to_categorical(y_train, 2)
    return x_train, y_train


def train_model(amount, x_train, y_train, model):
#    w,h=amount,amount
#    w,h=amount,amount-1
#    w,h=amount,1  #attributeerror, nonetype
#    w,h=amount,5  #attributeerror, nonetype
    w,h=100,5

    model.add(Convolution2D(amount, w, h, border_mode='same', 
                            input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(MaxPooling2D(pool_size=(w-1, h-1)))

    model.add(Convolution2D(amount, w, h, border_mode='same'))
    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # build the model and use sgd with default parameters
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    
    # train and test
    model.fit(x_train, y_train, batch_size=3, nb_epoch=3, show_accuracy=True,
              validation_split=0.1, shuffle=True)
    return model
    

if __name__ == '__main__':


    fileName="Data/x_train.csv"
    f=open(fileName)
    fileName="Data/y_train.csv"
    f2=open(fileName)
    
    # cnn structure
    model = Sequential()
    amount=100  # amount must be >= width
    for i in range(1,2):
        x_train,y_train = get_data(amount,i,f,f2)        
        model = train_model(amount, x_train, y_train, model)
