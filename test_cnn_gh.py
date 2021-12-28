# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 09:34:25 2021

@author: guohuan
"""
import os
import copy
import random
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
from mxnet import autograd
from sklearn import metrics
from mxnet import gluon, init
import matplotlib.pyplot as plt
from sklearn import preprocessing
from mxnet.gluon import loss as gloss
from sklearn.model_selection import train_test_split
import time
from collections import Counter

#%% Count
params_file = os.getcwd() + "/20211201/params/cnn_epoch100.mxnet"
def run_cnn(params_file):
    data = np.load(os.getcwd() + '/data/test_data_gh.npy')
    
#    Data_=[]
#    for i in range(len(data)):
#        Data_.append(preprocessing.minmax_scale(np.array(data[i][1:]).reshape(6,85),axis=1))
#    Data_scaled=np.array(Data_).reshape(len(Data_),6*85)
    
    net = mx.gluon.nn.HybridSequential()
    net.add( mx.gluon.nn.Conv2D(channels=32, kernel_size=(2,7), activation='relu'),
             mx.gluon.nn.MaxPool2D(pool_size=2, strides=1),
             mx.gluon.nn.Conv2D(channels=64, kernel_size=(2,3), activation='relu'),
             mx.gluon.nn.MaxPool2D(pool_size=2, strides=1),
             mx.gluon.nn.Conv2D(channels=128, kernel_size=(2,3), padding=1, activation='relu'),
             mx.gluon.nn.MaxPool2D(pool_size=2, strides=1),
             mx.gluon.nn.Dense(512, activation='relu'),
             mx.gluon.nn.Dropout(0.5),
             mx.gluon.nn.Dense(256, activation='relu'),
             mx.gluon.nn.Dropout(0.5),
             mx.gluon.nn.Dense(1, activation='sigmoid'))
    
    Data_scaled_nd=nd.array(data).reshape(len(data),1,6,85)
    print("The account of test dataset is:", len(Data_scaled_nd))
    net.load_parameters(params_file)
    net.hybridize()
    predict_score_cnn=net(Data_scaled_nd).asnumpy()
    
    np.save(os.getcwd() + '/20211201/predict_score_cnn.npy', predict_score_cnn)
    np.savetxt(os.getcwd() + '/20211201/CNN_distributuin_train.txt', predict_score_cnn, fmt='%f', delimiter=',')


def classify3(up, down, y_predict_score_cnn):
    
    num_fuzzy = 0
    y_predict_cnn=[]
    for i in range(len(y_predict_score_cnn)):
        if y_predict_score_cnn[i]>=up:
            y_predict_cnn.append(2)
        elif y_predict_score_cnn[i]<down:
            y_predict_cnn.append(0)
        elif (y_predict_score_cnn[i]>down) and (y_predict_score_cnn[i]<up):
            y_predict_cnn.append(1)
            num_fuzzy = num_fuzzy + 1
    return y_predict_cnn, num_fuzzy

def acc_all(predict_label, label):
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(len(label)):
        if predict_label[i]==0:
            if label[i] == 0:
                tn+=1
            elif label[i] == 1:
                fn+=1
        elif predict_label[i]==2:
            if label[i] == 0:
                fp+=1
            elif label[i] == 1:
                tp+=1
    a=tp+tn+fp+fn
    return (tp+tn)/a


y = np.load(os.getcwd() + '/data/test_label_gh.npy')
print(len(y))
run_cnn(params_file)
predict_score_cnn = np.load(os.getcwd() + '/20211201/predict_score_cnn.npy')

#fuzzy = [(0.1,0.9), (0.15,0.85), (0.2,0.8), (0.25,0.75), (0.3,0.7), (0.35,0.65), (0.4,0.6), (0.45,0.55)]
fuzzy = [(0.08,0.92), (0.16,0.84), (0.24,0.76), (0.32,0.68), (0.4,0.6)]

for pair in fuzzy:
    down = pair[0]
    up = pair[1]
    fuzzy_ = up - down
    print("up:", up, "down", down, "The persentage of fuzzy set:", fuzzy_)

    y_predict_cnn, num_fuzzy = classify3(up, down, predict_score_cnn)
    fuzzy_ratio = num_fuzzy / len(y)
    print("num_fuzzy",num_fuzzy, "fuzzy_ratio",fuzzy_ratio, "fuzzy_ratio",fuzzy_ratio)
    print(Counter(y_predict_cnn))
    acc = acc_all(y_predict_cnn, y)

    fpr, tpr, thresholds = metrics.roc_curve(y, predict_score_cnn, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=' (area = %0.2f)'%auc)
    print("acc:",acc, "auc:", auc)

