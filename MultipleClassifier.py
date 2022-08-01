import numpy as np
import os
#import csv

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
#from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score



if __name__== '__main__':

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
    

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)
    
    classifiers = [(KNeighborsClassifier(), "Nearest Neighbor"),
                   (LDA(), "Linear Discriminant Analysis"),
                   (SVC(probability = True), "Support Vector Machine"),
                   (RandomForestClassifier(), "Random Forest"),
                   (LogisticRegression(), "Logistic Regression")]
    accuracies = []
    for clf,name in classifiers:    
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)
        accuracy = roc_auc_score(y_test, y_pred_proba[:,1])
        accuracies.append(accuracy) 
        
        
    print('Accuracies KNN = %.4f, LDA = %.4f, SVC = %.4f, RF = %.4f and LR = %.4f'  % (accuracies[0], accuracies[1], accuracies[2], accuracies[3], accuracies[4])) 
    
#        csv_file=open("rf.csv","w")
#        csv_file.write("GeneId,Prediction\n")
#        i=1
#        for pred in y_pred_proba:
#    #        print(pred, pred[0], pred[1])
#            m = pred[1]
#            csv_file.write(str(i)+","+str(m)+"\n")
#            i=i+1
        