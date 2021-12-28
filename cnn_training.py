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
    print('CNN test_predict_label_count',len(predit_score))
    np.savetxt(os.getcwd() + '/20211201/result_combined/3/CNN_distributuin_test.txt', predit_score, fmt='%f', delimiter=',')
    np.savetxt(os.getcwd() + '/20211201/result_combined/3/roc_cnn_test_fpr', fpr, fmt='%f', delimiter=',')
    np.savetxt(os.getcwd() + '/20211201/result_combined/3/roc_cnn_test_tpr', tpr, fmt='%f', delimiter=',')
    
    
    
    testing_loss = test_l_sum / n
    
    return testing_loss, test_auc


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
    np.savetxt(os.getcwd() + '/20211201/result_combined/3/CNN_distributuin_valid.txt', predit_score, fmt='%f', delimiter=',')
    np.savetxt(os.getcwd() + '/20211201/result_combined/3/roc_cnn_valid_fpr', fpr, fmt='%f', delimiter=',')
    np.savetxt(os.getcwd() + '/20211201/result_combined/3/roc_cnn_valid_tpr', tpr, fmt='%f', delimiter=',')

   
        
    validating_loss = valid_l_sum / n
    
    return validating_loss, valid_auc
    
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
        
        validating_loss, valid_auc = validating_auc(net, valid_iter, loss, ctx)
        
        
        train_loss.append(training_loss)
        valid_loss.append(validating_loss)
        
        np.savetxt(os.getcwd() + '/20211201/result_combined/3/CNN_distributuin_train.txt', predit_score, fmt='%f', delimiter=',')

        
        
        print(epoch, training_loss, validating_loss, valid_auc)
        
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            
            if valid_auc > best_valid_auc:
                
                testing_loss, test_auc = testing_auc(net, test_iter, loss, ctx)
                
                
                
            net.save_parameters(os.getcwd() + "/20211201/params/cnn_epoch" + str(epoch) + ".mxnet")
            np.savetxt(os.getcwd() + '/20211201/result_combined/3/roc_cnn_train_fpr', fpr, fmt='%f', delimiter=',')
            np.savetxt(os.getcwd() + '/20211201/result_combined/3/roc_cnn_train_tpr', tpr, fmt='%f', delimiter=',')

    
    
    x = list(range(len(train_loss)))
    plt.plot(x, train_loss, 'b', x, valid_loss, 'r')
    print("The training_auc:", train_auc, "The validing_auc:", valid_auc, "The testing_auc:", test_auc)
    
    return train_loss, valid_loss, testing_loss

batch_size = 64


train_data = np.load(os.getcwd()+'/data/train_data_gh.npy')
train_label = np.load(os.getcwd()+'/data/train_label_gh.npy')

valid_data = np.load(os.getcwd()+'/data/valid_data_gh.npy')
valid_label = np.load(os.getcwd()+'/data/valid_label_gh.npy')

test_data = np.load(os.getcwd()+'/data/test_data_gh.npy')
test_label = np.load(os.getcwd()+'/data/test_label_gh.npy')

train_data, train_label = nd.array(train_data).reshape(len(train_data),1, 6, 85), nd.array(train_label)   
valid_data, valid_label = nd.array(valid_data).reshape(len(valid_data),1, 6, 85), nd.array(valid_label)   
test_data, test_label = nd.array(test_data).reshape(len(test_data),1, 6, 85), nd.array(test_label)   


data_iter_train = mx.io.NDArrayIter(train_data, label=train_label, batch_size=batch_size, shuffle=True, last_batch_handle='discard')
data_iter_test  = mx.io.NDArrayIter(test_data, label=test_label, batch_size=batch_size, shuffle=True, last_batch_handle='discard')
data_iter_valid  = mx.io.NDArrayIter(valid_data, label=valid_label, batch_size=batch_size, shuffle=True, last_batch_handle='discard')



"""
train_data = np.load(os.getcwd() + '/data/nonshuflle_x.npy')
train_label = np.load(os.getcwd() + '/data/nonshuffle_label.npy')

data1_ = []
for i in range(len(train_data)):
    data1_.append(preprocessing.minmax_scale(np.array(train_data[i]).reshape(6,85),axis=1))
train_data = np.array(data1_).reshape(len(data1_), 1, 6, 85)

data2_scaled = np.load(os.getcwd() + '/data/test_data-hqz.npy')
test_label = np.load(os.getcwd() + '/data/test_label-hqz.npy')

#data2_ = []
#for i in range(len(test_data)):
#    data2_.append(preprocessing.minmax_scale(np.array(test_data[i][1:]).reshape(6,85),axis=1))
#data2_scaled = np.array(data2_).reshape(len(data2_), 6*85)


test_data = nd.array(data2_scaled).reshape(len(data2_scaled), 1, 6, 85)

data_iter_train = mx.io.NDArrayIter(train_data, label=train_label, batch_size=batch_size, shuffle=True)
data_iter_test  = mx.io.NDArrayIter(test_data, label=test_label, batch_size=batch_size, shuffle=True)
"""

ctx = mx.gpu()
lr, num_epochs = 0.001, 101

"""
class Residual(nn.HybridBlock):  # 本类已保存在d2lzh包中方便以后使用
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.HybridSequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk

net = nn.HybridSequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))

net.add(resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2))

net.add(nn.Dropout(0.5),
        nn.Dense(256, activation='relu'), 
        nn.Dropout(0.5),
        nn.Dense(1, activation='sigmoid'))
"""

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

net.cast('float32')
net.collect_params().initialize(mx.init.Xavier(), ctx=mx.gpu())
net.hybridize()

time_start = time.time()
#optim = optimizer.Adam(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08,wd=1e-5)
trainer = gluon.Trainer(net.collect_params(), 'adam',optimizer_params={'learning_rate':lr,'wd':5e-4}) #{'wd':5e-4}
train_loss, valid_loss, testing_loss = train_ch5(net, data_iter_train, data_iter_valid, data_iter_test, batch_size, trainer, ctx, num_epochs)

time_end = time.time()
time_sum = time_end - time_start
print("CNN time:", time_sum)

np.savetxt(os.getcwd() + '/20211201/result_combined/3/cnn_trainingLosses.txt', train_loss, fmt='%f', delimiter=',')
np.savetxt(os.getcwd() + '/20211201/result_combined/3/cnn_validatingLosses.txt', valid_loss, fmt='%f', delimiter=',')
 
params_file = os.getcwd() + "/20211201/params/cnn_epoch100.mxnet"

##%% Count
#params_file = os.getcwd() + "/params/cnn_epoch100.mxnet"
#def run_cnn(params_file):
#    data = np.load('test_data_gh.npy')
#    Data_=[]
#    for i in range(len(data)):
#        Data_.append(preprocessing.minmax_scale(np.array(data[i][1:]).reshape(6,85),axis=1))
#    Data_scaled=np.array(Data_).reshape(len(Data_),6*85)
#    
#    net = mx.gluon.nn.HybridSequential()
#    net.add( mx.gluon.nn.Conv2D(channels=32, kernel_size=(2,7), activation='relu'),
#             mx.gluon.nn.MaxPool2D(pool_size=2, strides=1),
#             mx.gluon.nn.Conv2D(channels=64, kernel_size=(2,3), activation='relu'),
#             mx.gluon.nn.MaxPool2D(pool_size=2, strides=1),
#             mx.gluon.nn.Conv2D(channels=128, kernel_size=(2,3), padding=1, activation='relu'),
#             mx.gluon.nn.MaxPool2D(pool_size=2, strides=1),
#             mx.gluon.nn.Dense(512, activation='relu'),
#             mx.gluon.nn.Dropout(0.5),
#             mx.gluon.nn.Dense(256, activation='relu'),
#             mx.gluon.nn.Dropout(0.5),
#             mx.gluon.nn.Dense(1, activation='sigmoid'))
#    
#    Data_scaled_nd=nd.array(Data_scaled).reshape(len(Data_scaled),1,6,85)
#    net.load_parameters(params_file)
#    net.hybridize()
#    predict_score_cnn=net(Data_scaled_nd).asnumpy()
#    
#    np.save(os.getcwd() + 'predict_score_cnn.npy', predict_score_cnn)
#
#def classify3(up, down, y_predict_score_cnn):
#    y_predict_cnn=[]
#    for i in range(len(y_predict_score_cnn)):
#        if y_predict_score_cnn[i]>=up:
#            y_predict_cnn.append(2)
#        elif y_predict_score_cnn[i]<down:
#            y_predict_cnn.append(0)
#        elif (y_predict_score_cnn[i]>down) and (y_predict_score_cnn[i]<up):
#            y_predict_cnn.append(1)
#    return y_predict_cnn
#
#def acc_all(predict_label, label):
#    tp=0
#    tn=0
#    fp=0
#    fn=0
#    for i in range(len(label)):
#        if predict_label[i]==0:
#            if label[i] == 0:
#                tn+=1
#            elif label[i] == 1:
#                fn+=1
#        elif predict_label[i]==2:
#            if label[i] == 0:
#                fp+=1
#            elif label[i] == 1:
#                tp+=1
#    a=tp+tn+fp+fn
#    return (tp+tn)/a
#
#
#y = np.load('test_label_gh.npy')
#run_cnn(params_file)
#predict_score_cnn = np.load('predict_score_cnn.npy')
#up=0.9
#down=0.1
#y_predict_cnn = classify3(up, down, predict_score_cnn)
#print(Counter(y_predict_cnn))
#acc = acc_all(y_predict_cnn, y)
#print(acc)
#fpr, tpr, thresholds = metrics.roc_curve(y, predict_score_cnn, pos_label=1)
#auc = metrics.auc(fpr, tpr)
#plt.plot(fpr, tpr, label=' (area = %0.2f)'%auc)
#print(auc)
#
