# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt

labels = ['H3K4me3','H3K4me1','H3K36me3','H3K9me3','H3K27me3']

def draw_heatmap(rankings):
    sns.set()
    

    w, h = 3, 500;
    f = [[0 for x in range(w)] for y in range(h)] 
    print(rankings)
    i=0
    for rank in rankings:
        print(i,int(i/5)+1,labels[i%5],rank)
        ind=i%5
        dev=int(i/5)+1
        f[i][0]=labels[ind]
        f[i][1]=dev
        f[i][2]=rank
        i=i+1    
    data_long=DataFrame(f,columns=['HM_mod', 'index', 'ranking'])    
#    data_pivot = data_long.pivot("HM_mod", "index", "ranking")
    data_pivot = data_long.pivot("index", "HM_mod", "ranking")
    


    
    # get the tick label font size
    fontsize_pt = plt.rcParams['ytick.labelsize']
    dpi = 72.27
    
    # comput the matrix height in points and inches
    matrix_height_pt = fontsize_pt * data_pivot.shape[0]
    matrix_height_in = matrix_height_pt / dpi
    
    # compute the required figure height 
    top_margin = 0.04  # in percentage of the figure height
    bottom_margin = 0.04 # in percentage of the figure height
    figure_height = matrix_height_in / (1 - top_margin - bottom_margin) +10
    
    
    # build the figure instance with the desired height
    fig, ax = plt.subplots(
            figsize=(5,figure_height), 
            gridspec_kw=dict(top=1-top_margin, bottom=bottom_margin))
    

    # Draw a heatmap with the numeric values in each cell
#    sns.heatmap(data_pivot, annot=True, fmt="d", linewidths=.4, xticklabels=2, cbar_kws={"orientation": "horizontal"})
#    sns.heatmap(data_pivot, annot=True, fmt="d", xticklabels=2, linewidths=.4, cbar_kws={"orientation": "vertical"})
        
#    f, ax = plt.subplots(figsize=(20, 100))    

    sns.heatmap(data_pivot, annot=True, fmt="d",cbar_kws={"orientation": "vertical"})
#    sns.plt.yticks(rotation=0) 
    
#    sns.plt.show()
    
    
    # save the figure
    plt.savefig('test.png')


def getData():
    data_path = "." # This folder holds the csv files
    
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
    return X_train, X_test, y_train


def rfe(x_train, x_test, y_train):
#    X_train, X_test, Y_train, Y_test=   train_test_split(x_train, y_train, train_size=0.8)
    X_train, X_test, Y_train, Y_test=   train_test_split(x_train, y_train, train_size=0.8)
    
    # Select features
    rfecv = RFECV(estimator = LogisticRegression(), step = 25, cv = 10)
    print("X_train: ", X_train.shape)
    print("y_train: ", Y_train.shape)
    
#    x_train_pia=np.transpose(X_train.reshape(500,-1))
    X_train=np.transpose(X_train.reshape(500,-1))
    print(X_train)


    rfecv.fit(X_train, Y_train.flatten())
    print("\nNumber of selected features:", rfecv.n_features_)
    print("grid scores: ",rfecv.grid_scores_) #
    print("ranking: ",rfecv.ranking_) #
    print("support: ",rfecv.support_) #
    draw_heatmap(rfecv.ranking_)
    
#    # Train the whole training set with the selected features
    lr1 = LogisticRegression()
    lr1.fit(X_train[:, rfecv.support_], Y_train.flatten())
#    
    X_test=np.transpose(X_test.reshape(500,-1))

#    y_test=lr1.predict_proba(X_test[:, rfecv.support_])
    # Performance on the test set
    score_lr1 = accuracy_score(Y_test.flatten(), lr1.predict(X_test[:, rfecv.support_]))
    print("RFECV accuracy score on the test set:", score_lr1)
        

def l1(x_train, x_test, y_train):
    X_train, X_test, Y_train, Y_test=   train_test_split(x_train, y_train, train_size=0.8)    

    # Select features
    print("X_train: ", X_train.shape)
    print("y_train: ", Y_train.shape)
    
    X_train=np.transpose(X_train.reshape(500,-1))
    
    # L1 reg, 10-fold CV
    parameters =     {
        'penalty': ['l1'],
        'C': np.logspace(-4, 5, 20)
    } 
    clf = GridSearchCV(estimator=LogisticRegression(), 
                                        param_grid = parameters,
                                        cv = 10, 
                                        n_jobs=-1)
                                        
    clf.fit(X_train, Y_train.flatten())
    best_params = clf.best_params_    
    print("\nBest parameters:", best_params)
    print("grid_scores: ",clf.grid_scores_)

#    draw_heatmap(clf.grid_scores_)
    
    logreg = LogisticRegression(penalty = best_params['penalty'], C = best_params['C'])
    logreg.fit(X_train, y_train.flatten())
    print("Number of selected features:", np.count_nonzero(logreg.coef_))
    
    # Performance on the test set
    score_logreg = accuracy_score(Y_test.flatten(), logreg.predict(X_test))
    draw_heatmap(logreg.grid_scores_)

    print("L1-regularized LR accuracy:", score_logreg)
    

if __name__ == '__main__':
    
    X_train, X_test, y_train = getData()
    rfe(X_train,X_test,y_train)
#    l1(X_train, X_test, y_train)