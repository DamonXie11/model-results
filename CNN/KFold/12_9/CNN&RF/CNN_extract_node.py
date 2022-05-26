import keras.regularizers
import pandas as pd
import csv as csv
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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


def step_decay(epoch):
    initial_lrate = 0.0015
    # drop = 0.5
    # epochs_drop = 10.0
    # lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return initial_lrate

def CNN_all(file_name):
    data_path_and_labels = pd.read_csv(
        r'C:\Users\13508\Desktop\Software_fault_prediction-main\Deep_learning_regression\datasets\mapped data' + file_name + '.csv')

    # if(data_path_and_labels['filename']):

    # data = data_path_and_labels

    data = data_path_and_labels.drop(['filename'],axis=1)

    for i in range(len(data['bugs'])):
        if data['bugs'][i] > 1:
            data['bugs'][i] = 1

    F1 = 0
    kf = KFold(n_splits=10, shuffle=True, random_state= 256)
    for train_index, test_index in kf.split(data):
        train = data.loc[train_index]
        test = data.loc[test_index]

        # train, test = train_test_split(data, test_size=0.2)

        Y_train = pd.DataFrame(train['bugs'], columns=['bugs'])
        Y_test = pd.DataFrame(test['bugs'], columns=['bugs'])

        train = train.drop('bugs', axis=1)
        test = test.drop('bugs', axis=1)

        X_train = train

        scaler = MinMaxScaler()
        MinMaxScaler(copy=True, feature_range=(0, 1))
        X_train = MinMaxScaler().fit_transform(X_train)
        X_test = MinMaxScaler().fit_transform(test)
        Y_train = MinMaxScaler().fit_transform(Y_train)

        img_rows, img_cols = 1, X_train.shape[1]

        X_train1 = X_train.reshape(X_train.shape[0], img_cols, 1)
        X_test1 = X_test.reshape(X_test.shape[0], img_cols, 1)
        input_shape = (img_cols, 1)

        ros = RandomOverSampler(random_state=0)
        X_train1, Y_train = ros.fit_resample(X_train, Y_train)
        # Counter(Y_train['bugs'])

        # Building the model
        model = Sequential()

        # add model layers
        model.add(Embedding(output_dim=32, input_dim=X_train.shape[0], input_length=img_cols,
                            embeddings_regularizer=keras.regularizers.l2(0.1)))
        model.add(Conv1D(100, kernel_size=2, strides=2, activation='relu', kernel_regularizer=l2(0.0001),
                         kernel_initializer="he_normal", input_shape=input_shape))
        # model.add(MaxPool1D(pool_size=8))
        model.add(Conv1D(64, kernel_size=2, strides=2, activation='relu', kernel_initializer="he_normal"))
        model.add(MaxPool1D(pool_size=8))
        # model.add(Conv1D(16, kernel_size=2, strides=2, activation='relu', kernel_initializer="he_normal"))
        # model.add(MaxPool1D(pool_size= 5 ))

        model.add(Flatten())
        model.add(Dense(50, activation='relu', kernel_initializer="he_normal"))
        # model.add(Dropout(0.5))
        # model.add(Dense(8, activation='relu', kernel_initializer="he_normal"))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid', kernel_initializer="glorot_normal"))

        # compile model using mse as the loss function
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), metric_F1score])

        model.summary()

        reduce_lr = LearningRateScheduler(step_decay)

        early_stopping = EarlyStopping(monitor='loss', patience=50)
        print("X_train", X_train1.shape)
        print("Y_train", Y_train.shape)
        history = model.fit(X_train1, Y_train, epochs=100, batch_size=256, validation_data=(X_test1, Y_test), shuffle= True,
                            callbacks=[reduce_lr])

        score = model.evaluate(X_test1, Y_test, verbose=1)
        f1_list.append(score[4])
        F1 += score[4]
        f = open(
            r'C:\Users\13508\Desktop\Software_fault_prediction-main\Deep_learning_regression\datasets\predict_data'  +file_name+ '_result.csv',
            'a', encoding='utf-8', newline='' "")

        csv_writer = csv.writer(f)

        # csv_writer.writerow(["Y","Y_predict"])

        csv_writer.writerow(score)
        f.close()

        # plt.plot(history.history['metric_F1score'])
        # plt.plot(history.history['val_metric_F1score'])
        # plt.title('Train History')
        # plt.ylabel('F1-score')
        # plt.xlabel('Epoch')
        # plt.legend(['train', 'validation'], loc='upper right')
        # plt.savefig(
        #     r'C:\Users\13508\Desktop\Software_fault_prediction-main\Deep_learning_regression\datasets\result_pic\a'+file_name+'_F1_' + str(
        #         i) + '.png')
        # plt.show()
        #
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('Train History')
        # plt.ylabel('loss')
        # plt.xlabel('Epoch')
        # plt.legend(['train', 'validation'], loc='upper right')
        # plt.savefig(
        #     r'C:\Users\13508\Desktop\Software_fault_prediction-main\Deep_learning_regression\datasets\result_pic\a' +file_name+'_loss_' + str(
        #         i) + '.png')
        plt.show()

        prediction_y = model.predict(X_test1)

        prediction_y_round = np.rint(prediction_y)

    return F1/10

if __name__ == '__main__':
    # file_name_list = ["\ivy-1.1", "\ivy-1.4", "\ivy-2.0", "\jedit-3.2", "\jedit-4.0", "\jedit-4.1", "\jedit-4.2", "\jedit-4.3","\log4j-1.0", "\log4j-1.1", "\log4j-1.2", "\lucene-2.0", "\lucene-2.2", "\lucene-2.4", "\pbeans-1.0", "\pbeans-2.0", "\poi-1.5", "\poi-2.0", "\poi-2.5", "\poi-3.0", "\synapse-1.0", "\synapse-1.1", "\synapse-1.2", r"\velocity-1.4", r"\velocity-1.5", r"\velocity-1.6", r"\xalan-2.4", r"\xalan-2.5", r"\xalan-2.6", r"\xerces-1.2", r"\xerces-1.3", r"\xerces-init"]
    file_name_list = [r"\xerces-1.2", r"\xerces-1.3", r"\xerces-init"]
    f1_list =[]

    for item in file_name_list:
        try:
            for i in range(0, 1):
                if i != 1:
                    res = CNN_all(item)
        except:
            pass


    # avr = 0
    # for item in f1_list:
    #     print(item)
    #     avr += item
    # avr /= 5
    print(res)


