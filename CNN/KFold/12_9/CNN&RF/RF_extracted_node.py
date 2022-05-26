import keras.regularizers
import pandas as pd
import csv as csv
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,Conv2D,Conv1D,Flatten,MaxPool2D,MaxPool1D,Embedding
from keras.metrics import TrueNegatives,TruePositives,FalseNegatives,FalsePositives
import keras.backend as K
from keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.regularizers import l2
from collections import Counter
from keras.callbacks import LearningRateScheduler
import math


#F1-score评价指标
def metric_F1score(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1score=2*precision*recall/(precision+recall)
    return F1score

def cal_metric_F_measure(origin, predict):
    length = len(origin)
    len1 = len(predict)
    values = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'F1': 0}
    if(length != len1):
        print("cal_metric_F_measure 输入长度不一致")
        return
    TP, FP, FN, TN = 0, 0, 0, 0
    for item_origin in origin:
        for item_predict in predict:
            if item_origin == item_predict and item_predict  == 1:
                TP += 1
            elif item_origin == 1 and item_predict == 0:
                FN += 1
            elif item_origin == 0 and item_predict == 1:
                FP += 1
            elif item_origin == item_predict and item_predict == 0:
                TN += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1score = 2 * precision * recall / (precision + recall)

    return F1score


def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

def CNN_all(file_name):

    data_path_and_labels = pd.read_csv(
        r'C:\Users\13508\Desktop\Software_fault_prediction-main\Deep_learning_regression\datasets\extracted data\node_12_9' + file_name + '.csv')

    data = data_path_and_labels.drop(['filename'], axis=1)
    # data = data_path_and_labels


    for i in range(len(data['bugs'])):
        if data['bugs'][i] > 1:
            data['bugs'][i] = 1

    kf = StratifiedKFold(n_splits=10,shuffle=True)
    F_measure = 0

    train = data
    # X_train, X_test = X.loc[train_index], X.loc[test_index]
    # y_train, y_test = Y.loc[train_index], Y.loc[test_index]

    # print(X_train)

    # train, test = train_test_split(data, test_size=0.2)

    Y_train = pd.DataFrame(train['bugs'], columns=['bugs'])

    train = train.drop('bugs', axis=1)

    X_train = train

    scaler = MinMaxScaler()
    MinMaxScaler(copy=True, feature_range=(0, 1))
    X_train = MinMaxScaler().fit_transform(X_train)
    Y_train = MinMaxScaler().fit_transform(Y_train)
    # print(X_train, Y_train)

    img_rows, img_cols = 1, X_train.shape[1]

    # X_train1 = X_train.reshape(X_train.shape[0], img_cols, 1)
    # X_test1 = X_test.reshape(X_test.shape[0], img_cols, 1)
    input_shape = (img_cols, 1)


    ros = RandomOverSampler(random_state=0)
    X_train1, Y_train1 = ros.fit_resample(X_train, Y_train)
    # X_train1, Y_train1 = X_train, Y_train
    print(X_train.shape)
    kf = StratifiedKFold(n_splits=10, shuffle=True)
    F_measure = 0
    for train_index, test_index in kf.split(X_train1, Y_train1):
        # print('train_index', train_index)
        # print('X_train', X_train.shape)
        X_train, X_test = X_train1[train_index], X_train1[test_index]
        y_train, y_test = Y_train1[train_index], Y_train1[test_index]
        # train = data.loc[train_index]
        # test = data.loc[test_index]
        # print(X_train, y_train)

        # Counter(Y_train['bugs'])
        # 随机森林
        clf2 = RandomForestClassifier(criterion="entropy", n_estimators=50, max_depth=None, min_samples_split=2,
                                      random_state=0)
        clf2.fit(X_train, y_train)

        pre_Y_test = clf2.predict(X_test)

        F1 = cal_metric_F_measure(y_test, pre_Y_test)
        # print("y_test",y_test)
        # print("pre_y_test", pre_Y_test)
        F_measure += F1

    return F_measure/10

if __name__ == '__main__':
    file_name_list = [ "\ivy-1.1", "\ivy-1.4", "\ivy-2.0", "\jedit-3.2", "\jedit-4.0", "\jedit-4.1", "\jedit-4.2", "\jedit-4.3","\log4j-1.0", "\log4j-1.1", "\log4j-1.2", "\lucene-2.0", "\lucene-2.2", "\lucene-2.4", "\pbeans-1.0", "\pbeans-2.0", "\poi-1.5", "\poi-2.0", "\poi-2.5", "\poi-3.0", "\synapse-1.0", "\synapse-1.1", "\synapse-1.2", r"\velocity-1.4", r"\velocity-1.5", r"\velocity-1.6", r"\xalan-2.4", r"\xalan-2.5", r"\xalan-2.6", r"\xerces-1.2", r"\xerces-1.3", r"\xerces-init"]

    # file_name_list = [r'\camel-1.2']
    f1_list =[]

    file_res = open(
        r'C:\Users\13508\Desktop\Software_fault_prediction-main\Deep_learning_regression\datasets\extracted data\RF_res.csv',
        'w+')
    try:
        for item in file_name_list:
            for i in range(0, 1):
                if i != 1:
                    res = CNN_all(item)
                    file_res.write(item + ',' + str(res))
                    file_res.write('\n')
                    print("average_F1:", res)
        file_res.close()
    except:
        pass




