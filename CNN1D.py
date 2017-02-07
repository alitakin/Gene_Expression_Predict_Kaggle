'''Trains a simple convnet on the REMC dataset.'''
import numpy as np
import os
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

import matplotlib.pyplot as plt

#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils


if __name__== '__main__':
    data_path = "C:\\Users\\Z RY\\Documents\\TUT\\Competetion\\Gene_Expression_Predict_Kaggle" # This folder holds the csv files
    
    # load csv files. We use np.loadtxt. Delimiter is ","
    # and the text-only header row will be skipped.   
    
    print("Loading data...")
    x_train = np.loadtxt(data_path + os.sep + "x_train.csv", 
                         delimiter = ",", skiprows = 1)
    x_test  = np.loadtxt(data_path + os.sep + "x_test.csv", 
                         delimiter = ",", skiprows = 1)    
    y_train = np.loadtxt(data_path + os.sep + "y_train.csv", 
                         delimiter = ",", skiprows = 1)
    
    print ("All files loaded. Preprocessing...")
    
    # remove the first column(Id)
    x_train = x_train[:,1:] 
    x_test  = x_test[:,1:]   
    y_train = y_train[:,1:] 
    
    # Every 100 rows correspond to one gene.
    # Extract all 100-row-blocks into a list using np.split.
    num_genes_train = x_train.shape[0] / 100
    num_genes_test  = x_test.shape[0] / 100
    
    print("Train / test data has %d / %d genes." % (num_genes_train, num_genes_test))
    x_train = np.split(x_train, num_genes_train)
    x_test  = np.split(x_test, num_genes_test)
    
    # Reshape by raveling each 100x5 array into a 500-length vector
    #x_train = [g.ravel() for g in x_train]
    #x_test  = [g.ravel() for g in x_test]
    
    # convert data from list to array
    X_train = np.array(x_train)
    y_train = np.array(y_train)
    X_test  = np.array(x_test)
    y_train = np.ravel(y_train)
    
    # Preprocessing

#    X_train = preprocessing.scale(X_train) 
#    X_test = preprocessing.scale(X_test) 
    
    
    batch_size = 20
    num_classes = 2
    num_epochs = 50
    
    # input data dimensions
    data_shape = (100, 5)
    # number of convolutional filters to use
    num_featmaps = 20
    # size of pooling area for max pooling
    pool = 2
    # convolution window size
    window_size = 5
    
    # the data, shuffled and split between train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
    

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, num_classes)
    Y_test = np_utils.to_categorical(y_test, num_classes)
    
    
#   Prepare model 
    model = Sequential()
#    Layaer 1 : needs input_shape as well
    model.add(Convolution1D(num_featmaps, window_size,
                            border_mode = 'valid',
                            input_shape = data_shape, activation = 'relu'))

#    Layer 2: 
    model.add(Convolution1D(num_featmaps, window_size, activation = 'relu', border_mode = 'same'))
    model.add(MaxPooling1D(pool_length = pool))
    model.add(Dropout(0.25))
    
    model.add(Convolution1D(num_featmaps, window_size, activation = 'relu', border_mode = 'same'))
    model.add(MaxPooling1D(pool_length = pool))
    model.add(Dropout(0.25))
    
    model.add(Convolution1D(num_featmaps, window_size, activation = 'relu', border_mode = 'same'))
    model.add(MaxPooling1D(pool_length = pool))
    model.add(Dropout(0.25))
    
    model.add(Convolution1D(num_featmaps, window_size, activation = 'relu', border_mode = 'same'))
    model.add(MaxPooling1D(pool_length = pool))
    model.add(Dropout(0.25))
    
    model.add(Convolution1D(num_featmaps, window_size, activation = 'relu', border_mode = 'same'))
    model.add(MaxPooling1D(pool_length = pool))
    model.add(Dropout(0.25))
    
    model.add(Convolution1D(num_featmaps, window_size, activation = 'relu', border_mode = 'same'))
    model.add(MaxPooling1D(pool_length = pool))
    model.add(Dropout(0.25))
    
    
#    Layer 3 : dense layer with 128 nodes
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    
#    Last layer: producing 2 outputs
    model.add(Dense(num_classes, activation = 'softmax'))   
    
#    Compile and run
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=num_epochs, validation_data=(X_test, Y_test))
    
    score = model.evaluate(X_test, Y_test, batch_size = batch_size)
    
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
#    model.save("CNN1D.h5")
    
    y_pred = model.predict_classes(X_test)
    y_pred = np_utils.to_categorical(y_pred, num_classes)
#    y_pred = model.predict_classes(X_test, batch_size = batch_size, verbose = 1)

    acuracy = roc_auc_score(Y_test, y_pred)
    print('Accuraacy is %.4f : ' % (acuracy))
    
    
    csv_file=open("CNN1D.csv","w")
    csv_file.write("GeneId,Prediction\n")
    i=1
    for pred in y_pred:
        m = pred[1]
        csv_file.write(str(i)+","+str(m)+"\n")
        i=i+1