import numpy as np
import os
from keras.callbacks import ModelCheckpoint
#import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils

#%%
def modelTrain(Xtrain, ytrain , Xtest, num_classes):
    d = dict()
#    History_list=[]
    batch_size = 20
    #num_classes = 2
    num_epochs = 100
    # input data dimensions
    data_shape = (100, 5)
    # number of convolutional filters to use
    num_featmaps = 20
    # size of pooling area for max pooling
    pool = 2
    # convolution window size
    window_size = 5
    
    print('Xtrain shape:', Xtrain.shape)
    print(Xtrain.shape[0], 'train samples')
    print(Xtest.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    Ytrain = np_utils.to_categorical(ytrain, num_classes)
#    Y_test = np_utils.to_categorical(y_test, num_classes)
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
    
#%%    
#    Layer 3 : dense layer with 128 nodes
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    
#    Last layer: producing 2 outputs
    model.add(Dense(num_classes, activation = 'softmax'))   
#%%    
#    Compile and run
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath="weights.best.hdf5"
    checkpointer = ModelCheckpoint(filepath = filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    history = model.fit(Xtrain, Ytrain, batch_size=batch_size, nb_epoch=num_epochs, callbacks = [checkpointer])

    model.summary()
#    model.save("CNN1D.h5")
#%%    
    model.load_weights("weights.best.hdf5")   
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    d['y_class'] = model.predict_classes(Xtest)
    d['y_prob'] = model.predict(Xtest, batch_size = batch_size, verbose = 1)
#    y_pred = np_utils.to_categorical(y_pred, num_classes)
   
#%%
#
#    print(history.history.keys())
#    CNN_hist = history.history
#    list_CNN_hist = [v for v in CNN_hist.values()]
#    CNN_history = np.array(list_CNN_hist)
#    History_list.append(CNN_history)
#    csv_History = open("CNN_history.csv","w")
#    csv_History.write("vall_loss,val_acc,loss,acc\n") 
#    for i in range(num_epochs):
#        csv_History.write(str(CNN_history[0,i])+","+str(CNN_history[1,i])+","+str(CNN_history[2,i])+","+str(CNN_history[3,i])+"\n")
#    
    
    return d


#%%
def get_csv(y_probability):
    csv_file=open("CNN_testFeedback.csv","w")
    csv_file.write("GeneId,Prediction\n")
    i=1
    for pred in y_probability:
        m = pred[1]
        csv_file.write(str(i)+","+str(m)+"\n")
        i=i+1 
       
#%%    
    
    
if __name__ == "__main__":
    
    randomseed = 0
    np.random.seed(randomseed)
    
    num_classes = 2
        
    data_path = "./" # This folder holds the csv files
    
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

        # convert data from list to array
    X_train = np.array(x_train)
    y_train = np.array(y_train)
    X_test  = np.array(x_test)
    y_train = np.ravel(y_train)     
    
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
#%%    
    ## run for 10 times to get a mean of probabilities
    predictions = []
    for i in range(10):
        print("This is the %03d th run." %i) 
        d =modelTrain(X_train, y_train , X_test, num_classes)
        predictions.append(d['y_prob'])  
    d['y_prob'] = np.mean(predictions, axis = 0)  
        
#%%    Find those test samples with confidence > 0.1 
  
    for i in range(5):
        X_new = []    
        y_new = []
    #    relabelWeight = 10,
        relabelThr = 0.1
        
        p = d['y_prob'][:,1]
        
        for idx in range(X_test.shape[0]):
        
            # "Confidence" is the distance from 0.5
            # w contains the number of times each test sample is
            # included in the new training data.
            # w is zero for samples below confidence threshold.
            
            confidence = np.abs(p[idx] - 0.5)
            if confidence > relabelThr:
                w = int(10 * confidence)
            
                X_new += [X_test[idx, ...]] * w
                y_new += [np.round(p[idx])] * w
        
        if i == 0:                   
            X_new_train=np.concatenate((X_train,X_new),axis=0)
            y_new_train=np.concatenate(( y_train, y_new),axis=0)
        else:
            X_new_train=np.array(X_new)
            y_new_train=np.array(y_new)

        d = modelTrain(X_new_train, y_new_train, X_test, num_classes)
#        y_pred_class = d['y_class']
    y_proba = d['y_prob']
    get_csv(y_proba)
        
    print ("job is completed") 