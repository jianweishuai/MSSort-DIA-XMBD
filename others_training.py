# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 22:16:49 2021

@author: hqz
"""

import time
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
from sklearn import preprocessing
from mxnet.gluon import loss as gloss
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from sklearn.svm import SVC
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier


train_data = np.load(os.getcwd()+'/data/train_data_gh.npy')
train_label = np.load(os.getcwd()+'/data/train_label_gh.npy')

valid_data = np.load(os.getcwd()+'/data/valid_data_gh.npy')
valid_label = np.load(os.getcwd()+'/data/valid_label_gh.npy')

test_data  = np.load(os.getcwd()+'/data/test_data_gh.npy')
test_label = np.load(os.getcwd()+'/data/test_label_gh.npy')

train_data = train_data.reshape(len(train_data), 6*85)
valid_data = valid_data.reshape(len(valid_data), 6*85)
test_data = test_data.reshape(len(test_data), 6*85)
#%%

#clf_SVM = joblib.load("clf_svm.m")
#test_svm = clf_SVM.predict_proba(test_data)
#test_s = test_svm[:, 1]


#%%

time_start = time.time()
"""=============================Train SVM============================ """
clf_svm = SVC(C=0.5, gamma=0.1, kernel='rbf', probability=True)


clf_svm_model = clf_svm.fit(train_data, train_label)
time1 = time.time()
time_fit = time1 - time_start
print("fit time:", time_fit)
joblib.dump(clf_svm_model, "clf_svm.m")
    
clf_s = joblib.load("clf_svm.m")

train_predict_SVM_label = clf_s.decision_function(train_data)    


fpr_train_SVM, tpr_train_SVM , thresholds = metrics.roc_curve(train_label, train_predict_SVM_label, pos_label=1)

train_auc = metrics.auc(fpr_train_SVM, tpr_train_SVM)


valid_predict_SVM_label = clf_s.decision_function(valid_data)

fpr_valid_SVM, tpr_valid_SVM , thresholds = metrics.roc_curve(valid_label, valid_predict_SVM_label, pos_label=1)

valid_auc = metrics.auc(fpr_valid_SVM, tpr_valid_SVM)

test_predict_SVM_label = clf_s.decision_function(test_data)

test_svm = clf_s.predict_proba(test_data)
test_s = test_svm[:, 1]

np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/SVM/SVM_test_probablity.txt', test_s, fmt='%f', delimiter=',')


fpr_test_SVM, tpr_test_SVM , thresholds = metrics.roc_curve(test_label, test_predict_SVM_label, pos_label=1)

test_auc = metrics.auc(fpr_test_SVM, tpr_test_SVM)
print('test_auc',test_auc,'train_auc',train_auc,'valid_auc',valid_auc)

np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/SVM_train_fpr.txt', fpr_train_SVM, fmt='%f', delimiter=',')
np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/SVM_train_tpr.txt', tpr_train_SVM, fmt='%f', delimiter=',')

np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/SVM_valid_fpr.txt', fpr_valid_SVM, fmt='%f', delimiter=',')
np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/SVM_valid_tpr.txt', tpr_valid_SVM, fmt='%f', delimiter=',')

np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/SVM_test_fpr.txt', fpr_test_SVM, fmt='%f', delimiter=',')
np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/SVM_test_tpr.txt', tpr_test_SVM, fmt='%f', delimiter=',')

np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/SVM_distributuin_train.txt', train_predict_SVM_label, fmt='%f', delimiter=',')
np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/SVM_distributuin_valid.txt', valid_predict_SVM_label, fmt='%f', delimiter=',')
np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/SVM_distributuin_test.txt', test_s, fmt='%f', delimiter=',')


time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('SVM Time:', time_sum)

#%%

"""=============================Train RandomForest============================ """
clf_rf = RandomForestClassifier(n_estimators=250)
time_start = time.time()

clf_rand = clf_rf.fit(train_data, train_label)
joblib.dump(clf_rand, "clf_rand.m")

clf_r = joblib.load("clf_rand.m")

train_predict_rf_prob = clf_r.predict_proba(train_data)
train_rf = train_predict_rf_prob[:, 1]

fpr_train_RandomForest, tpr_train_RandomForest , thresholds = metrics.roc_curve(train_label, train_rf, pos_label=1)
train_auc = metrics.auc(fpr_train_RandomForest, tpr_train_RandomForest)


valid_predict_rf_prob = clf_r.predict_proba(valid_data)
valid_rf = valid_predict_rf_prob[:, 1]

fpr_valid_RandomForest, tpr_valid_RandomForest , thresholds = metrics.roc_curve(valid_label, valid_rf, pos_label=1)
valid_auc = metrics.auc(fpr_valid_RandomForest, tpr_valid_RandomForest)


test_predict_rf_prob = clf_r.predict_proba(test_data)
test_rf = test_predict_rf_prob[:, 1]

fpr_test_RandomForest, tpr_test_RandomForest , thresholds = metrics.roc_curve(test_label, test_rf, pos_label=1)
test_auc_rand = metrics.auc(fpr_test_RandomForest, tpr_test_RandomForest)

np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/SVM/random_test_predic_label.txt', test_rf, fmt='%f', delimiter=',')


#print('Randforest test_auc:', test_auc_rand)
print('Randforest test_auc',test_auc_rand,'train_auc',train_auc,'valid_auc',valid_auc)

np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/RandomForest_train_fpr.txt', fpr_train_RandomForest, fmt='%f', delimiter=',')
np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/RandomForest_train_tpr.txt', tpr_train_RandomForest, fmt='%f', delimiter=',')

np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/RandomForest_valid_fpr.txt', fpr_valid_RandomForest, fmt='%f', delimiter=',')
np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/RandomForest_valid_tpr.txt', tpr_valid_RandomForest, fmt='%f', delimiter=',')

np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/RandomForest_test_fpr.txt', fpr_test_RandomForest, fmt='%f', delimiter=',')
np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/RandomForest_test_tpr.txt', tpr_test_RandomForest, fmt='%f', delimiter=',')

time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('RandomForest Time:', time_sum)


np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/RandomForest_distributuin_train.txt', train_rf, fmt='%f', delimiter=',')
np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/RandomForest_distributuin_valid.txt', valid_rf, fmt='%f', delimiter=',')
np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/RandomForest_distributuin_test.txt', test_rf, fmt='%f', delimiter=',')



#%%
time_start = time.time()
"""=============================Test Spearmanr============================ """
all_spearmanr = []

test_data = test_data.reshape(len(test_data), 6, 85)

for i in range(len(test_label)):
    
    sp_list = []
    
    for j in range(5):
        for k in range(5):
            if k != j:
                r, _ = spearmanr(test_data[i][j], test_data[i][k])
                sp_list.append(r)
            
    all_spearmanr.append(np.mean(sp_list)) #画all_spearmanr的直方图

np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/PearsonrSpearmanr/Spearsonr_com.txt', all_spearmanr, fmt='%f', delimiter=',')

#np.savetxt('D:/guohuan/lym/submission/PearsonrSpearmanr/Spearsonr_indi.txt', all_spearmanr, fmt='%f', delimiter=',')
time_end1 = time.time()  # 记录结束时间
time_sum1 = time_end1 - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('Spearman Time:', time_sum1)

#%%
"""=============================Test Pearson============================ """
all_pearsonr = []

test_data = test_data.reshape(len(test_data), 6, 85)

for i in range(len(test_label)):
    
    p_list = []
    
    for j in range(5):
        for k in range(5):
            if k != j:
                r, _ = pearsonr(test_data[i][j], test_data[i][k])##
                p_list.append(r)
            
        
    all_pearsonr.append(np.mean(p_list)) #画all_pearsonr的直方图

np.savetxt('D:/guohuan/lym/submission/20211201/result_combined/3/PearsonrSpearmanr/pearsonr_com.txt', all_pearsonr, fmt='%f', delimiter=',')
#np.savetxt('D:/guohuan/lym/submission/PearsonrSpearmanr/pearsonr_indi.txt', all_pearsonr, fmt='%f', delimiter=',')
time_end2 = time.time()  # 记录结束时间
time_sum2 = time_end2 - time_end1  # 计算的时间差为程序的执行时间，单位为秒/s
print('Pearson Time:', time_sum2)



