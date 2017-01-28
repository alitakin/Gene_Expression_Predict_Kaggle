import numpy as np
import os
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D


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

    print("Train / test data has %d / %d genes." % \
          (num_genes_train, num_genes_test))
    x_train = np.split(x_train, num_genes_train)
    x_test  = np.split(x_test, num_genes_test)

    # Reshape by raveling each 100x5 array into a 500-length vector
    x_train = [g.ravel() for g in x_train]
    x_test  = [g.ravel() for g in x_test]
    
    # convert data from list to array
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test  = np.array(x_test)
    y_train = np.ravel(y_train)
    
    # Now x_train should be 15485 x 500 and x_test 3871 x 500.
    # y_train is 15485-long vector.
    
    print("x_train shape is %s" % str(x_train.shape))    
    print("y_train shape is %s" % str(y_train.shape))
    print("x_test shape is %s" % str(x_test.shape))
    
    print('Data preprocessing done...')
    
    print("Next steps FOR YOU:")
    print("-" * 30)
    print("1. Define a classifier using sklearn")
    print("2. Assess its accuracy using cross-validation (optional)")
    print("3. Fine tune the parameters and return to 2 until happy (optional)")
    print("4. Create submission file. Should be similar to y_train.csv.")
    print("5. Submit at kaggle.com and sit back.")
     
    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=50)
    
	
	# These parameters should be modified.
    num_featmaps = 10 # This many filters per layer
    num_classes = 2 # Digits 0,1
    num_epochs = 50 # Show all samples 50 times
    w, h = 5, 5 # Conv window size

    
    model = Sequential()
    
    # Layer 1: needs input_shape as well.
	# input_shape needs to be corrected. 
    model.add(Convolution2D(num_featmaps, w, h, input_shape=(, , ), activation = 'relu'))
    # Layer 2:
    model.add(Convolution2D(num_featmaps, w, h, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
	
	# The following numbers can also be modified based on our problem:
    # Layer 3: dense layer with 128 nodes
    # Flatten() vectorizes the data:
    # 32x10x10 -> 3200
    # (10x10 instead of 14x14 due to border effect)
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    # Layer 4: Last layer producing 10 outputs.
    model.add(Dense(num_classes, activation='softmax'))
    # Compile and train
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    model.fit(X_train, y_train, nb_epoch=100)
    
    y_hat = model.predict(X_test)
#    y_hat = np.array(model.predict(X_test))
    p = model.predict_proba(X_test)
#    p = np.array(model.predict_proba(X_test))
    print(y_hat)
    print(p)
    
    y_true = np.array(y_test)

    acuracy = roc_auc_score(y_true, y_hat)
    print(acuracy)
#
