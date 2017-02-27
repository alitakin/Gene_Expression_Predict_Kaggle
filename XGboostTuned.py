import numpy as np
import os

import xgboost as xgb
import xgboost.sklearn as XGB
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

if __name__ == '__main__':
    data_path = "./"  # This folder holds the csv files

    # load csv files. We use np.loadtxt. Delimiter is ","
    # and the text-only header row will be skipped.

    print("Loading data...")
    x_train = np.loadtxt(data_path + os.sep + "x_train.csv",
                         delimiter=",", skiprows=1)
    x_test = np.loadtxt(data_path + os.sep + "x_test.csv",
                        delimiter=",", skiprows=1)
    y_train = np.loadtxt(data_path + os.sep + "y_train.csv",
                         delimiter=",", skiprows=1)

    print ("All files loaded. Preprocessing...")

    # remove the first column(Id)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]
    y_train = y_train[:, 1:]

    # Every 100 rows correspond to one gene.
    # Extract all 100-row-blocks into a list using np.split.
    num_genes_train = x_train.shape[0] / 100
    num_genes_test = x_test.shape[0] / 100

    print("Train / test data has %d / %d genes." % \
          (num_genes_train, num_genes_test))
    x_train = np.split(x_train, num_genes_train)
    x_test = np.split(x_test, num_genes_test)

    # Reshape by raveling each 100x5 array into a 500-length vector
    x_train = [g.ravel() for g in x_train]
    x_test = [g.ravel() for g in x_test]

    # convert data from list to array
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_train = np.ravel(y_train)

    # Now x_train should be 15485 x 500 and x_test 3871 x 500.
    # y_train is 15485-long vector.

    print("x_train shape is %s" % str(x_train.shape))
    print("y_train shape is %s" % str(y_train.shape))
    print("x_test shape is %s" % str(x_test.shape))

    print('Data preprocessing done...')

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

    # specify parameters via map
    # param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
    bst = xgb.XGBClassifier(learning_rate=0.1,
                            n_estimators=5000,
                            max_depth=5,
                            min_child_weight=1,
                            gamma=0.4,
                            subsample=0.9,
                            reg_alpha= 1e-10,
                            colsample_bytree=0.9,
                            objective='binary:logistic',
                            nthread=4,
                            scale_pos_weight=1,
                            seed=27)

    bst.fit(x_train, y_train)

    # make prediction

    y_pred = bst.predict(x_test)


    accuracy = roc_auc_score(y_test, y_pred)

    print("the accuurace for XGboost is: ", accuracy)

    # export the prediction probablities
    # y_predict_proba = bst.predict_proba(x_test)

    # csv_file = open("XGboost.csv", "w")
    # csv_file.write("GeneId,Prediction\n")
    # i = 1
    # for pred in y_predict_proba:
    #    m = pred[1]
    #    csv_file.write(str(i) + "," + str(m) + "\n")
    #    i = i + 1

    # csv_file.close()
    # print ("CSV file ready.")
    print("bst made!")

    param_test1 = {
        'max_depth': range(2, 10, 2),
        'min_child_weight': range(1, 6, 2)
    }
    param_test2 = {
        'max_depth': [3,4,5],
        'min_child_weight': [4,5,6]
    }
    param_test3 = {
        'gamma': [i / 10.0 for i in range(3, 9)]
    }
    param_test4 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    param_test6a = {
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
    }
    param_test6b = {
        'reg_alpha': [1e-10, 1e-7, 1e-5]
    }
    param_test7 = {
        'max_depth': [3, 4, 5],
        'min_child_weight': [4, 5, 6],
        'gamma': [i / 10.0 for i in range(3, 9)],
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)],
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
        'reg_alpha': [1e-10, 1e-7, 1e-5,1e-4,1e-03]

    }
    #gsearch1 = GridSearchCV(estimator=bst,
    #                        param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    #print("grid search cv made")
    #gsearch1.fit(x_train, y_train)

    #print("data trained")
    #print("best params: ", gsearch1.best_params_)
    #print("best score: ",gsearch1.best_score_)

    #max_depth = gsearch1.best_params_
    #print("max depth saved.")


