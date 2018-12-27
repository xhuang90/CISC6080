# Author: Xinhong Huang
# CISC 6080 Capstone
# feature selection


import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from sklearn.model_selection import cross_val_score


def get_data(df):

    # df = pd.read_csv(dataset)

    xDf = df.iloc[:, 0: len(df.columns) - 1]
    yDf = df.iloc[:, len(df.columns) - 1]

    X = np.array(xDf.values)
    Y = np.array(yDf.values)

    return X, Y


def drop_duration(df):

    df_new = df.drop(['duration'], axis = 1)

    return df_new


def pcc(x, y):

    PCC = stats.pearsonr(x, y)

    return PCC[0]


def normalize_data(X):

    normX = np.zeros((len(X), len(X[0])))
    for col in range(len(X[0])):
        m = np.mean(X[:, col])
        stdv = np.std(X[:, col])
        normX[:, col] = (X[:, col] - m) / float(stdv)

    return normX


def selected_features_data(X, sorted_feat, m):

    selected_feats_X = X[:, sorted_feat[0]]

    if m > 1:
        for i in range(1, m):
            selected_feats_X = np.vstack((selected_feats_X, (X[:, sorted_feat[i]])))
    else:
        selected_feats_X = np.vstack((selected_feats_X, np.zeros(len(X))))

    return selected_feats_X.T


def filter_method(X, y, sorted_feat):

    m_dict = {}

    for m in range(1, len(X[0]) + 1):

        sltd_x = selected_features_data(X, sorted_feat, m)
        clf = LogisticRegression()
        scores = cross_val_score(clf, sltd_x, y, cv = 10)
        m_dict[m] = np.mean(scores)

    optimal_accuracy = max(m_dict.values())

    print('Values m and their corresponding accuracies:')

    for m, acc in m_dict.items():
        print(m, ',', round(acc, 3))

    print()
    print('Value of m gives the highest 10-fold CV classification accuracy, and the value of this optimal accuracy:')

    for m, acc in m_dict.items():
        
        if acc == optimal_accuracy:
            
            print('m = ' + str(m), 'accuracy=' + format(acc, '.3f'))

            return m


def wrapper_method(X,y):

    feat_dict = {}   # empty set for selected features

    print(feat_dict)

    improve = True
    prev_acr = -1

    while improve:

        acr = np.zeros(len(X[0]))
        for feat in range(0,len(X[0])):
            sltd_lst = list(feat_dict.keys())
            sltd_lst.append(feat)

            sltd_x = selected_features_data(X, sltd_lst, len(sltd_lst))
            clf = LogisticRegression()
            scores = cross_val_score(clf, sltd_x, y, cv=10)

            acr[feat] = np.mean(scores)
        if len(feat_dict) > 0:
            for i in feat_dict.keys():
                acr[i] = -1

        idx_max = np.argmax(acr)

        if acr[idx_max] > prev_acr:

            feat_dict[idx_max] = format(acr[idx_max],'.3f')
            print(feat_dict)
        else:
            improve = False

        prev_acr = acr[idx_max]


if __name__ == '__main__':

    df = pd.read_csv('bank_sub1_ohc.csv')
    df_drop_dur = df.drop(['duration'], axis = 1)

    X, y = get_data(df_drop_dur)

    X = normalize_data(X)

    pearson_dict = {}
    att_lst = [att for att in df_drop_dur]

    for i in range(len(X[0])):
        pearson_dict[i] = abs(pcc(X[:, i], y))

    sorted_feat = sorted(pearson_dict, key = pearson_dict.get, reverse=True)
    print(sorted_feat)
    sorted_list = [(att_lst[feat], format(pearson_dict[feat], '2f'))
                   for feat in sorted(pearson_dict, key=pearson_dict.get, reverse=True)]
    att_sort = [tup[0] for tup in sorted_list]
    pcc_sort = [tup[1] for tup in sorted_list]

    print('Pearson Correlation Coefficient Results:')
    pcc_df = pd.DataFrame(
        {
            'Attribute Name': att_sort,
            'PCC abs value': pcc_sort,
        }
    )
    print(pcc_df)
    print()

    filter_start = time.time()
    filter_method(X, y, sorted_feat)
    filter_end = time.time()
    print('Running Time: ' + str(filter_end - filter_start) + 'sec.')
    print()

    wrapper_start = time.time()
    wrapper_method(X, y)
    wrapper_end = time.time()
    print('Running Time: ' + str(wrapper_end - wrapper_start) + 'sec.')