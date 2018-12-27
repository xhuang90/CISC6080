# Author: Xinhong Huang
# CISC 6080 Capstone
# Model

import pandas as pd
import numpy as np
import time
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


def get_data(dataset):

    df = pd.read_csv(dataset)

    xDf = df.iloc[:, 0: len(df.columns) - 1]
    yDf = df.iloc[:, len(df.columns) - 1]

    X = np.array(xDf.values)
    Y = np.array(yDf.values)

    return X, Y


def split_bal_data(dataset):

    df = pd.read_csv(dataset)

    # create training and  testing vars
    train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 0)
    # print(train_set.shape) (32950, 47)
    # print(test_set.shape) (8238, 47)

    # print(train_set['y'].value_counts())
    # 0 29229, 1 3721
    train_majority = train_set[train_set.y == 0]
    train_minority = train_set[train_set.y == 1]
    train_minority_upsampled = resample(train_minority, replace = True,\
                                        n_samples = 29229, random_state = 0)
    train_upsampled = pd.concat([train_majority, train_minority_upsampled])
    # print(train_upsampled.y.value_counts())

    train_upsampled.to_csv('bank_sub1_train_balanced.csv', index = False)
    test_set.to_csv('bank_sub1_test.csv', index = False)


# normalize train and test data using mean, std of train data
def normalize_data(X, T):

    normX = np.zeros((len(X),len(X[0])))
    normT = np.zeros((len(T),len(T[0])))
    for col in range(len(X[0])):
        m = np.mean(X[:,col])
        stdv = np.std(X[:,col])
        normX[:, col] = (X[:,col]-m)/float(stdv)
        normT[:, col] = (T[:,col]-m)/float(stdv)

    return normX, normT


def RF_param_select(x, y):

    # parameters tunning
    parameters={'n_estimators': [10,30,50,70], 'max_features': [None,'sqrt'], \
                'max_depth': [None,20,40], 'min_weight_fraction_leaf': [0,0.2]}

    RFclf = RandomForestClassifier(class_weight = 'balanced', random_state = 7)
    clf = GridSearchCV(RFclf,parameters, scoring = 'precision_macro',\
                       n_jobs = -1, cv = 5, return_train_score = True)
    clf.fit(x,y)

    result2 = pd.DataFrame(clf.cv_results_)
    result2 = result2[['mean_test_score','mean_train_score','param_max_depth',\
                     'param_max_features','param_min_weight_fraction_leaf','param_n_estimators','rank_test_score']]

    result3 = result2[(result2['mean_test_score'] > 0.5) \
                      & ((result2['mean_train_score'] - result2['mean_test_score']) < 0.2)]
    result3 = result3.sort_values(by = 'rank_test_score')

    print(result3)


def Rfclf(trainX, trainY, testX, testY):

    # build rf according to the best param
    RFclf = RandomForestClassifier(max_depth = 40, max_features = 'sqrt', \
                                 min_weight_fraction_leaf = 0, n_estimators = 10,
                                 class_weight = 'balanced', random_state = 0)

    # get train set result
    RFclf.fit(trainX, trainY)
    trainY_pred = RFclf.predict(trainX)
    print('Training Accuracy:', accuracy_score(trainY, trainY_pred))
    confusion_matrix_train = confusion_matrix(trainY, trainY_pred)
    print('Training Confusion Matrix: ')
    print(confusion_matrix_train)
    print()

    # get test set result
    testY_pred = RFclf.predict(testX)
    print('Testing Accuracy:', accuracy_score(testY, testY_pred))
    print('Precision Score: ', precision_score(testY, testY_pred))
    print('F1 Score: ', f1_score(testY, testY_pred))
    print('Recall Score: ', recall_score(testY, testY_pred))
    confusion_matrix_test = confusion_matrix(testY, testY_pred)
    print('Test Confusion Matrix: ')
    print(confusion_matrix_test)


def SVM_param_select(x, y):

    # parameters tunning
    parameters={'C': [1.0, 1.5, 2.0, 2.5], 'kernel': ['rbf']}

    SVMclf = SVC(class_weight = 'balanced', random_state = 7)
    clf = GridSearchCV(SVMclf, parameters, scoring = 'precision_macro',\
                       n_jobs = -1, cv = 5, return_train_score = True)
    clf.fit(x,y)

    result2 = pd.DataFrame(clf.cv_results_)
    result2 = result2[['mean_test_score','mean_train_score',\
                     'param_C','param_kernel','rank_test_score']]

    result3 = result2[(result2['mean_test_score'] > 0.5) \
                      & ((result2['mean_train_score'] - result2['mean_test_score']) < 0.2)]
    result3 = result3.sort_values(by = 'rank_test_score')

    print(result3)


def SVMclf(trainX, trainY, testX, testY):

    # build rf according to the best param
    SVMclf = SVC(C = 2.5, kernel = 'rbf', class_weight = 'balanced', random_state = 0)

    # get train set result
    SVMclf.fit(trainX, trainY)
    trainY_pred = SVMclf.predict(trainX)
    print('Training Accuracy:', accuracy_score(trainY, trainY_pred))
    confusion_matrix_train = confusion_matrix(trainY, trainY_pred)
    print('Training Confusion Matrix: ')
    print(confusion_matrix_train)
    print()

    # get test set result
    testY_pred = SVMclf.predict(testX)
    print('Testing Accuracy:', accuracy_score(testY, testY_pred))
    print('Precision Score: ', precision_score(testY, testY_pred))
    print('F1 Score: ', f1_score(testY, testY_pred))
    print('Recall Score: ', recall_score(testY, testY_pred))
    confusion_matrix_test = confusion_matrix(testY, testY_pred)
    print('Test Confusion Matrix: ')
    print(confusion_matrix_test)


def KNN_param_select(x, y):

    # parameters tunning
    parameters={'n_neighbors': [5, 9, 13, 15, 17, 21, 25], \
                'weights': ['uniform', 'distance'],\
                'metric': ['minkowski', 'manhattan']}

    KNNclf = KNeighborsClassifier(p = 2, metric_params = None)
    clf = GridSearchCV(KNNclf, parameters, scoring = 'precision_macro',\
                       n_jobs = -1, cv = 5, return_train_score = True)
    clf.fit(x,y)

    result2 = pd.DataFrame(clf.cv_results_)
    result2 = result2[['mean_test_score', 'mean_train_score', \
                     'param_n_neighbors', 'param_weights', \
                        'param_metric', 'rank_test_score']]

    result3 = result2[(result2['mean_test_score'] > 0.5) \
                      & ((result2['mean_train_score'] - result2['mean_test_score']) < 0.2)]
    result3 = result3.sort_values(by = 'rank_test_score')

    print(result3)


def KNNclf(trainX, trainY, testX, testY):

    # build rf according to the best param
    KNNclf = KNeighborsClassifier(n_neighbors = 5, weights = 'distance', \
                                  algorithm = 'ball_tree', metric = 'manhattan', \
                                  leaf_size = 20, p = 2, metric_params = None)

    # get train set result
    KNNclf.fit(trainX, trainY)
    trainY_pred = KNNclf.predict(trainX)
    print('Training Accuracy:', accuracy_score(trainY, trainY_pred))
    confusion_matrix_train = confusion_matrix(trainY, trainY_pred)
    print('Training Confusion Matrix: ')
    print(confusion_matrix_train)
    print()
    
    # get test set result
    testY_pred = KNNclf.predict(testX)
    print('Testing Accuracy:', accuracy_score(testY, testY_pred))
    print('Precision Score: ', precision_score(testY, testY_pred))
    print('F1 Score: ', f1_score(testY, testY_pred))
    print('Recall Score: ', recall_score(testY, testY_pred))
    confusion_matrix_test = confusion_matrix(testY, testY_pred)
    print('Test Confusion Matrix: ')
    print(confusion_matrix_test)


def NNclf(trainX, trainY, testX, testY):

    # build rf according to the best param
    NNclf = MLPClassifier(solver = 'sgd', activation = 'relu', alpha = 1e-4,
                    hidden_layer_sizes = (27,27), random_state = 1,
                    max_iter = 300, learning_rate_init = 0.001)

    # get train set result
    NNclf.fit(trainX, trainY)
    trainY_pred = NNclf.predict(trainX)
    print('Training Accuracy:', accuracy_score(trainY, trainY_pred))
    confusion_matrix_train = confusion_matrix(trainY, trainY_pred)
    print('Training Confusion Matrix: ')
    print(confusion_matrix_train)
    print()

    # get test set result
    testY_pred = NNclf.predict(testX)
    print('Testing Accuracy:', accuracy_score(testY, testY_pred))
    print('Precision Score: ', precision_score(testY, testY_pred))
    print('F1 Score: ', f1_score(testY, testY_pred))
    print('Recall Score: ', recall_score(testY, testY_pred))
    confusion_matrix_test = confusion_matrix(testY, testY_pred)
    print('Test Confusion Matrix: ')
    print(confusion_matrix_test)


def LGS_param_select(x, y):

    # parameters tunning
    parameters={'penalty': ['l1', 'l2'],\
                'C': [1.0, 1.5, 2.0, 2.5]}

    LGSclf = LogisticRegression(fit_intercept = True, random_state = 0)
    clf = GridSearchCV(LGSclf, parameters, scoring = 'precision_macro',\
                       n_jobs = -1, cv = 5, return_train_score = True)
    clf.fit(x,y)

    result2 = pd.DataFrame(clf.cv_results_)
    result2 = result2[['mean_test_score', 'mean_train_score', \
                     'param_penalty', 'param_C', 'rank_test_score']]

    result3 = result2[(result2['mean_test_score'] > 0.5) \
                      & ((result2['mean_train_score'] - result2['mean_test_score']) < 0.2)]
    result3 = result3.sort_values(by = 'rank_test_score')

    print(result3)


def LGSclf(trainX, trainY, testX, testY):

    # build rf according to the best param
    LGSclf = LogisticRegression(C = 1.0, penalty = 'l2', \
                                fit_intercept=True, random_state = 0)

    # get train set result
    LGSclf.fit(trainX, trainY)
    trainY_pred = LGSclf.predict(trainX)
    print('Training Accuracy:', accuracy_score(trainY, trainY_pred))
    confusion_matrix_train = confusion_matrix(trainY, trainY_pred)
    print('Training Confusion Matrix: ')
    print(confusion_matrix_train)
    print()

    # get test set result
    testY_pred = LGSclf.predict(testX)
    print('Testing Accuracy:', accuracy_score(testY, testY_pred))
    print('Precision Score: ', precision_score(testY, testY_pred))
    print('F1 Score: ', f1_score(testY, testY_pred))
    print('Recall Score: ', recall_score(testY, testY_pred))
    confusion_matrix_test = confusion_matrix(testY, testY_pred)
    print('Test Confusion Matrix: ')
    print(confusion_matrix_test)


if __name__ == '__main__':

    # split_bal_data('bank_sub1_ohc.csv')

    trainX_org, trainY = get_data('bank_sub1_train_balanced.csv')
    testX_org, testY = get_data('bank_sub1_test.csv')

    # nolmalization
    trainX, testX = normalize_data(trainX_org, testX_org)

    # Model1: Random Forest

    RF_param_select(trainX, trainY)

    print('***** Results of Random Forest *****')
    rf_start = time.time()
    Rfclf(trainX, trainY, testX, testY)
    rf_end = time.time()
    print()
    print('Running Time: ' + str(rf_end - rf_start) + 'sec')
    print('************************************')
    print()
    print()


    # Model2: SVM

    SVM_param_select(trainX, trainY)

    print('***** Results of Support Vector Machine *****')
    svm_start = time.time()
    SVMclf(trainX, trainY, testX, testY)
    svm_end = time.time()
    print()
    print('Running Time: ' + str(svm_end - svm_start) + 'sec')
    print('*********************************************')
    print()
    print()


    # Model3: KNN

    KNN_param_select(trainX, trainY)

    print('***** Results of K Nearest Neighbor *****')
    knn_start = time.time()
    KNNclf(trainX, trainY, testX, testY)
    knn_end = time.time()
    print()
    print('Running Time: ' + str(knn_end - knn_start) + 'sec')
    print('*****************************************')
    print()
    print()

    

    # Model4: Neural Network

    print('******* Results of Neural Network *******')
    nn_start = time.time()
    NNclf(trainX, trainY, testX, testY)
    nn_end = time.time()
    print()
    print('Running Time: ' + str(nn_end - nn_start) + 'sec')
    print('*****************************************')
    print()
    print()

    
    # Model5: Logistic Regression
    
    LGS_param_select(trainX, trainY)
    
    print('******* Results of Logistic Regression *******')
    lgs_start = time.time()
    LGSclf(trainX, trainY, testX, testY)
    lgs_end = time.time()
    print()
    print('Running Time: ' + str(lgs_end - lgs_start) + 'sec')
    print('**********************************************')
    print()
    print()


