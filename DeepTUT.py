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


def get_data():
    fileName="Data/x_train.csv"
    f=open(fileName)
    fileName="Data/y_train.csv"
    f2=open(fileName)

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
    samples=w*h
    x_train=np.random.random((samples, w, h, 32))
    x_train=np.array(x_train)
    print(x_train.shape)

#    for line in f2:
#        line = line.split('\n')[0]
#        prob = line.split(",")[1]
#        print(prob)
#        y_train.append([prob])
#        if (i%amount==0): break
#        i=i+1


    for i in range(0,samples):
        y_train.append([randint(0,1)])
    y_train = np.array(y_train)
    print(y_train.shape)
    y_train = np_utils.to_categorical(y_train, 2)
    return x_train, y_train




def train_model(x_train, y_train, model):
    w,h=100,5
    samples=w*h  # samples must be >= width

    # if tf, channels, rows, columns
    model.add(Convolution2D(samples, w, h, border_mode='same', 
                            input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(MaxPooling2D(pool_size=(w-1, h-1)))

    model.add(Convolution2D(samples, w, h, border_mode='same'))
    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=["accuracy"])
    
    model.fit(x_train, y_train, batch_size=3, nb_epoch=3,
              validation_split=0.1, shuffle=True)
    return model
    

if __name__ == '__main__':

    # cnn structure
    model = Sequential()
    for i in range(1,2):
        x_train,y_train = get_data()        
        model = train_model(x_train, y_train, model)
        #lots of data - if saving is needed
