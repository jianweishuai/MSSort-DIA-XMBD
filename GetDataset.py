# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 09:16:28 2021

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
from sklearn import preprocessing
from mxnet.gluon import loss as gloss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

comb_data = np.load(os.getcwd()+'/data/comb_data.npy')
comb_label = np.load(os.getcwd()+'/data/comb_label.npy')

X_train, X_test, Y_train, Y_test = train_test_split(comb_data, comb_label, stratify=comb_label, test_size=0.2, random_state=5)

X_train1, valid_data, Y_train1, valid_label = train_test_split(X_train, Y_train, stratify=Y_train, test_size=0.25, random_state=5)

np.save(os.getcwd()+'/data/train_data_gh.npy', X_train1)
np.save(os.getcwd()+'/data/train_label_gh.npy', Y_train1)

np.save(os.getcwd()+'/data/valid_data_gh.npy', valid_data)
np.save(os.getcwd()+'/data/valid_label_gh.npy', valid_label)

np.save(os.getcwd()+'/data/test_data_gh.npy', X_test)
np.save(os.getcwd()+'/data/test_label_gh.npy', Y_test)