import numpy as np
import os
#import csv

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score




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
    
    
#    col_max = np.amax(x_train, axis = 0)
#    row_max = np.amax(x_train, axis = 1)
#    whole_max = np.amax(x_train)
    
#    x_train = preprocessing.scale(x_train) 
#    x_test = preprocessing.scale(x_test) 
#    
#    print('Data preprocessing done...')
#    
#    
#    
#
#    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)
#    C_range = 10.0 ** np.arange(-6, 1)
#    clf = LogisticRegression()
#    for C in C_range:
#        for penalty in ["l1", "l2"]:
#            clf.C = C
#            clf.penalty = penalty           
#            clf.fit(X_train, y_train)
#            y_pred = clf.predict(X_test)
#            scores = cross_val_score(clf, X_test, y_test, cv = 5)
#            accuracy = roc_auc_score(y_test, y_pred)
#            y_pred_proba = clf.predict_proba(X_test)
#            print ("Accuracy for C = %.2e and penalty = %s is %.5f" % (C, penalty, accuracy))    
#            print(scores)
#            
##        csv_file=open("LogReg.csv","w")
##        csv_file.write("GeneId,Prediction\n")
##        i=1
##        for pred in y_pred_proba:
##            m = pred[1]
##            csv_file.write(str(i)+","+str(m)+"\n")
##            i=i+1
#        