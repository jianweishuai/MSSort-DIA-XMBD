# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 21:44:35 2021

@author: hqz
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
from sklearn import preprocessing
from mxnet.gluon import loss as gloss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

def testing_auc(net, test_iter, loss, ctx):
    
    test_iter.reset()
    n = 0
    test_l_sum = 0.0
    
    predit_score = []
    target_label = []
	
    for batch in test_iter:
            
        X = batch.data[0].as_in_context(ctx)
        y = batch.label[0].as_in_context(ctx)
    
        y_hat = net(X)
        l = loss(y_hat, y)
        
        n += 1
        test_l_sum += l.mean().asscalar()
        
        target_label.extend(y.asnumpy().tolist())
        predit_score.extend(y_hat.asnumpy().tolist())
    
    fpr, tpr, thresholds = metrics.roc_curve(target_label, predit_score, pos_label=1)
    test_auc = metrics.auc(fpr, tpr)
    print('RNN test_predict_label_count',len(predit_score))
    np.savetxt(os.getcwd() + '/20211201/result_combined/3/RNN_distributuin_test.txt', predit_score, fmt='%f', delimiter=',')
    np.savetxt(os.getcwd() + '/20211201/result_combined/3/roc_rnn_test_fpr', fpr, fmt='%f', delimiter=',')
    np.savetxt(os.getcwd() + '/20211201/result_combined/3/roc_rnn_test_tpr', tpr, fmt='%f', delimiter=',')
    
    
    
    testing_loss = test_l_sum / n
    
    return testing_loss, test_auc, fpr, tpr


def validating_auc(net, valid_iter, loss, ctx):
    
    n = 0
    valid_l_sum = 0.0
    
    predit_score = []
    target_label = []
	
    for batch in valid_iter:
            
        X = batch.data[0].as_in_context(ctx)
        y = batch.label[0].as_in_context(ctx)
    
        y_hat = net(X)
        l = loss(y_hat, y)
        
        n += 1
        valid_l_sum += l.mean().asscalar()
        
        target_label.extend(y.asnumpy().tolist())
        predit_score.extend(y_hat.asnumpy().tolist())
    
    fpr, tpr, thresholds = metrics.roc_curve(target_label, predit_score, pos_label=1)
    valid_auc = metrics.auc(fpr, tpr)
    np.savetxt(os.getcwd() + '/20211201/result_combined/3/RNN_distributuin_valid.txt', predit_score, fmt='%f', delimiter=',')
    np.savetxt(os.getcwd() + '/20211201/result_combined/3/roc_rnn_valid_fpr', fpr, fmt='%f', delimiter=',')
    np.savetxt(os.getcwd() + '/20211201/result_combined/3/roc_rnn_valid_tpr', tpr, fmt='%f', delimiter=',')
    
        
    validating_loss = valid_l_sum / n
    
    return validating_loss, valid_auc, fpr, tpr

class Model(gluon.nn.Block):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.rnn = gluon.rnn.RNN(64, 1, dropout=0.5, layout="NTC")
        self.out = gluon.nn.Dense(1, activation='sigmoid')
 
    def forward(self, x):
        f1 = self.rnn(x)[:,-1,:]
		#print(f1.shape)
        return(self.out(f1))

def train_ch5(net, train_iter, valid_iter, test_iter, batch_size, trainer, ctx, num_epochs):
    
    loss = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    
    train_loss=[]
    valid_loss=[]
    best_valid_auc = 0.0
    
    for epoch in range(num_epochs):
        
        n = 0.0
        train_l_sum = 0.0
        
        train_iter.reset()
        valid_iter.reset()
        
        
        predit_score = []
        target_label = []
        
        for batch in train_iter:
            
            X = batch.data[0].as_in_context(ctx)
            y = batch.label[0].as_in_context(ctx)
            
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            
            l.backward()        #计算梯度
            trainer.step(batch_size)        #更新参数
            train_l_sum += l.mean().asscalar()        #.asscalar()将NDArray转换为标量
            n += 1
            
            target_label.extend(y.asnumpy().tolist())
            predit_score.extend(y_hat.asnumpy().tolist())
        
        fpr, tpr, thresholds = metrics.roc_curve(target_label, predit_score, pos_label=1)
        train_auc = metrics.auc(fpr, tpr)
        
        training_loss = train_l_sum / n
        
        validating_loss, valid_auc, fpr_valid, tpr_valid = validating_auc(net, valid_iter, loss, ctx)
        
        
        train_loss.append(training_loss)
        valid_loss.append(validating_loss)
        
        np.savetxt(os.getcwd() + '/20211201/result_combined/3/RNN_distributuin_train.txt', predit_score, fmt='%f', delimiter=',')

        
        
        print(epoch, training_loss, validating_loss, valid_auc)
        
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            
            if valid_auc > best_valid_auc:
                
                testing_loss, test_auc, fpr_test, tpr_test = testing_auc(net, test_iter, loss, ctx)
                
            net.save_parameters(os.getcwd() + "/20211201/params/rnn_epoch" + str(epoch) + ".mxnet")
            np.savetxt(os.getcwd() + '/20211201/result_combined/3/roc_rnn_train_fpr', fpr, fmt='%f', delimiter=',')
            np.savetxt(os.getcwd() + '/20211201/result_combined/3/roc_rnn_train_tpr', tpr, fmt='%f', delimiter=',')
    
#    plt.figure(0)
    x = list(range(len(train_loss)))
    plt.plot(x, train_loss, 'b', x, valid_loss, 'r')
#    plt.figure(1)
#    plt.plot(fpr, tpr, 'b',fpr_valid, tpr_valid, 'r', fpr_test, tpr_test, 'g')
    print("The training_auc:", train_auc, "The validing_auc:", valid_auc, "The testing_auc:", test_auc)
    
    
    return train_loss, valid_loss, testing_loss, fpr, tpr

batch_size = 256

train_data = np.load(os.getcwd()+'/data/train_data_gh.npy')
train_label = np.load(os.getcwd()+'/data/train_label_gh.npy')

valid_data = np.load(os.getcwd()+'/data/valid_data_gh.npy')
valid_label = np.load(os.getcwd()+'/data/valid_label_gh.npy')

test_data = np.load(os.getcwd()+'/data/test_data_gh.npy')
test_label = np.load(os.getcwd()+'/data/test_label_gh.npy')

train_data, train_label = nd.array(train_data).reshape(len(train_data), 6, 85), nd.array(train_label)   
valid_data, valid_label = nd.array(valid_data).reshape(len(valid_data), 6, 85), nd.array(valid_label)   
test_data, test_label = nd.array(test_data).reshape(len(test_data), 6, 85), nd.array(test_label)   

data_iter_train = mx.io.NDArrayIter(train_data, label=train_label, batch_size=batch_size, shuffle=True, last_batch_handle='discard')
data_iter_test  = mx.io.NDArrayIter(test_data, label=test_label, batch_size=batch_size, shuffle=True, last_batch_handle='discard')
data_iter_valid  = mx.io.NDArrayIter(valid_data, label=valid_label, batch_size=batch_size, shuffle=True, last_batch_handle='discard')


ctx = mx.gpu()
lr, num_epochs = 0.001, 101

net = Model()

net.collect_params().initialize(mx.init.Normal(sigma=0.01), ctx=mx.gpu())
net.cast('float32')

time_start = time.time()
#optim = optimizer.Adam(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08,wd=1e-5)
trainer = gluon.Trainer(net.collect_params(), 'adam',optimizer_params={'learning_rate':lr,'wd':5e-4}) #{'wd':5e-4}
train_loss, valid_loss, testing_loss, fpr_train, tpr_train = train_ch5(net, data_iter_train, data_iter_valid, data_iter_test, batch_size, trainer, ctx, num_epochs)

time_end = time.time()
time_sum = time_end - time_start
print("RNN time:", time_sum)

np.savetxt(os.getcwd() + '/20211201/result_combined/3/rnn_trainingLosses.txt', train_loss, fmt='%f', delimiter=',')
np.savetxt(os.getcwd() + '/20211201/result_combined/3/rnn_validatingLosses.txt', valid_loss, fmt='%f', delimiter=',')
