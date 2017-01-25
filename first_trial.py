import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
#    plt.figure()
#    plt.title(title)
#    if ylim is not None:
#        plt.ylim(*ylim)
#    plt.xlabel("Training examples")
#    plt.ylabel("Score")
#    train_sizes, train_scores, test_scores = learning_curve(
#        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#    train_scores_mean = np.mean(train_scores, axis=1)
#    train_scores_std = np.std(train_scores, axis=1)
#    test_scores_mean = np.mean(test_scores, axis=1)
#    test_scores_std = np.std(test_scores, axis=1)
#    plt.grid()
#
#    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                     train_scores_mean + train_scores_std, alpha=0.1,
#                     color="r")
#    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
#    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#             label="Training score")
#    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#             label="Cross-validation score")
#
#    plt.legend(loc="best")
#    return plt
#    
#    


# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 17:07:19 2017

@author: lingyu, hehu
"""
# Example of loading the data.

import os
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

if __name__== '__main__':

    data_path = "C:\\Users\\Z RY\\Documents\\TUT\\Competetion" # This folder holds the csv files

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
    
#    x_zero = X[y == 0, :]
#    x_one = x_train[y == 1, :]
#    
#    plt.figure()
#    plt.plot(X_zero[:, 0], X_zero[:, 1], 'ro')
#    plt.plot(X_one[:, 0], X_one[:, 1], 'bo')
#    plt.show()
#    
    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=50)
    
#    model = LogisticRegression(penalty = "l1", C = 0.1)
    
    model = KNeighborsClassifier(n_neighbors=5)
    
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
#    y_hat = np.array(model.predict(X_test))
    p = model.predict_proba(X_test)
#    p = np.array(model.predict_proba(X_test))
    print(y_hat)
    print(p)
    
    y_true = np.array(y_test)

    acuracy = roc_auc_score(y_true, y_hat)
    print(acuracy)

#    from sklearn.svm import SVC
#    estimator = SVC(kernel='linear')
#    from sklearn.cross_validation import ShuffleSplit
#    cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)
#    
#    from sklearn.grid_search import GridSearchCV
#    gammas = np.logspace(-6, -1, 10)
#    classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(gamma=gammas))
#    classifier.fit(X_train, y_train)
#    
#    from sklearn.learning_curve import learning_curve
#    title = 'Learning Curves (SVM, linear kernel, $\gamma=%.6f$)' %classifier.best_estimator_.gamma
#    estimator = SVC(kernel='linear', gamma=classifier.best_estimator_.gamma)
#    plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
#    plt.show()
#    
#    classifier.score(X_test, y_test)    
#    from sklearn.cross_validation import cross_val_score
#    cross_val_score(classifier, X_train, y_train)

  

