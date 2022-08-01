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
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score


total_row_count=15485
row_count=10
x_fileName="Data/x_train.csv"
y_fileName="Data/y_train.csv"
x_testfileName="Data/x_test.csv"


def getX(x_file,counter=0, isBreaking=1):
    # data has to be 4D: sample_id, color_channel, y, X


    f=open(x_file) 
    f.seek(counter*row_count)
    if (isBreaking==0):
        count = int(sum(1 for line in f)/100)
    else: count=row_count 
    x_train=np.zeros((count,5,100,1))

    ex_id=""    
    i=0
    k=0
    for row in f:
        row=row.split("\n")[0]
        items = row.split(",")
        tmp_id=items[0]
        #header checks
        if (tmp_id=="GeneId"): continue
        try:int(tmp_id)
        except ValueError: continue

        if (tmp_id!=ex_id and ex_id!=""):
            i=i+1
            k=0
        if (i>row_count-1 and isBreaking==1):break

        j=0
        for item in items[1:]:
            x_train[i][j][k][0]=item
            j=j+1

        k=k+1
        ex_id=tmp_id
    return x_train
    
def getY(y_file):
    if (y_file==""):
        y_train=np.zeros((row_count,2))
        return y_train
    y_train=[]
    y_train=np.genfromtxt(y_fileName, delimiter=",",skip_header=1)
    y_train=y_train[:row_count]
    y_train = np.array(y_train)
    print(y_train.shape)
    return y_train

def get_data(x_file, y_file, counter):
    return getX(x_file,counter), getY(y_file)    


    
def add_layers(model):
    w,h=100,5
    samples=w*h  # samples must be >= width

    model.add(Convolution2D(samples, w, h, border_mode='same', 
                            input_shape=(5,100,1))) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,  2)))
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



def showAcc(x_train,y_train):
    # the data, shuffled and split between train and test sets
    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.4)
    nb_classes=2


    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
#    Y_train = np_utils.to_categorical(y_train, nb_classes)
    print(y_test.shape)
#    y_test = y_test[:,[1]].astype(int)
#    print(y_test)
#    Y_test = np_utils.to_categorical(y_test, nb_classes)
#    print(Y_test)
    
    
    y_pred = model.predict_classes(X_test)
#    y_pred = np_utils.to_categorical(y_pred[:,[1]], nb_classes)
#    y_pred = np_utils.to_categorical(y_pred.astype(int), nb_classes)
    y_pred = np_utils.to_categorical(y_pred, nb_classes)
#    y_pred = model.predict_classes(X_test, batch_size = batch_size, verbose = 1)
#    print(Y_test)
    print(y_pred)

    accuracy = roc_auc_score(y_test, y_pred)
    print('Accuracy is %.4f : ' % (accuracy))
    #lots of data - if saving is needed, save the model



def predict(model):
    x_test=getX(x_testfileName,1,isBreaking=0)
    print(x_test.shape)
#    y_test=getY("")
    y_test = model.predict_classes(x_test)
    print(y_test.shape)
    print(y_test)

#    batch_size=10
#    score = model.evaluate(x_test, y_test, batch_size = batch_size, verbose = 1)
#    
#    print('Test score:', score[0])
#    print('Test accuracy:', score[1])
#    
#    nb_classes=2
#    y_pred = model.predict_classes(x_test)
#    y_pred = np_utils.to_categorical(y_pred, nb_classes)
##    y_pred = model.predict_classes(X_test, batch_size = batch_size, verbose = 1)
#
#    accuracy = roc_auc_score(y_test, y_pred)
#    print('Accuracy is %.4f : ' % (accuracy))
#    return y_pred


def writePred(y_pred):
    csv_file=open("deeptut.csv","w")
    csv_file.write("GeneId,Prediction\n")
    i=1
    for pred in y_pred:
        m = pred[1]
        csv_file.write(str(i)+","+str(m)+"\n")
        i=i+1


if __name__ == '__main__':

    # cnn structure
    model = Sequential()
    add_layers(model)
    limit=int(total_row_count/row_count)
    
#    for counter in range(limit):
    for counter in range(1):
        x_train,y_train = get_data(x_fileName, y_fileName, counter) 
        train_model(x_train, y_train, model)
#        showAcc(x_train,y_train)
    #predict
    model.summary()
    y_pred = predict(model)
    if (len(y_pred)>0):
        writePred(y_pred)
