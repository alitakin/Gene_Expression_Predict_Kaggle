# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 15:11:33 2017

@author: pia
"""

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dropout, Flatten, Dense,Activation

from keras.models import Sequential
#from keras.utils import np_utils
import numpy as np

total_row_count=15485
row_count=100
x_fileName="Data/x_train.csv"
y_fileName="Data/y_train.csv"


def get_data(counter):
    
    # data has to be 4D: sample_id, color_channel, y, X
    x_train=np.zeros((row_count,5,100,1))
    y_train=[]
    f=open(x_fileName) 
    f.seek(counter*row_count)

    ex_id=""    
    i=0
    k=0
    for row in f:
        if (i>row_count-1):break
        if (i>100):break
        items = row.split(",")
        tmp_id=items[0]
        if (tmp_id=="GeneId"): continue #header row

        j=0
        for item in items[1:]:
            x_train[i][j][k][0]=item
            j=j+1
#        print(row+ " "+items[0])
        k=k+1
        if (tmp_id!=ex_id and ex_id!=""):
            i=i+1
            k=0
        ex_id=tmp_id


    y_train=np.genfromtxt(y_fileName, delimiter=",",skip_header=1)
    y_train=y_train[:row_count]
    y_train = np.array(y_train)
    print(y_train.shape)
#    y_train = np_utils.to_categorical(y_train, 2)
    return x_train, y_train


    
def add_layers(model):
    w,h=100,5
    samples=w*h  # samples must be >= width

    model.add(Convolution2D(samples, w, h, border_mode='same', 
                            input_shape=(5,100,1)))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,  2)))
#    model.add(MaxPooling2D(pool_size=(w-1, h-1)))

    model.add(Convolution2D(samples, w, h, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))


def train_model(x_train, y_train, model):
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=["accuracy"])    
#    model.fit(x_train, y_train, batch_size=3, nb_epoch=3,
#              validation_split=0.1, shuffle=True)
    model.fit(x_train, y_train, batch_size=10, nb_epoch=1,
              validation_split=0.1, shuffle=True)

if __name__ == '__main__':

    # cnn structure
    model = Sequential()
    add_layers(model)
    limit=int(total_row_count/row_count)
    
    for counter in range(limit):
        x_train,y_train = get_data(counter)   
        train_model(x_train, y_train, model)
        #lots of data - if saving is needed, save the model

    #predict