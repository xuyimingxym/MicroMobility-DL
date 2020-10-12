#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from math import sqrt
get_ipython().run_line_magic('matplotlib', 'inline')

import scipy.sparse as sp
from scipy.sparse import linalg

import seaborn as sns
sns.set()

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

from scipy.optimize import minimize
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import math
import numpy.linalg as la
from sklearn.svm import SVR


# In[14]:


df = pd.read_csv("GridData/GD.csv")
sc = pd.read_csv("GridData/Street_Connectivity.csv")
df = df.drop("pos_code",axis=1)
sc = sc.drop("pos_code",axis=1)


# In[15]:


df_removez =df[(df.T!= 0).any()]
df_removez =df_removez.T
df_removez


# In[18]:


First_half=df_removez.iloc[:,:350]
Sec_half=df_removez.iloc[:,350:]#df_removez.columns.get_loc(41):df_removez.columns.get_loc(550)]
First_half.index = np.array(range(df_removez.shape[0]))
Sec_half.index = np.array(range(df_removez.shape[0]))
plt.figure(figsize=(25,20))
plt.subplot(2,1,1)
plt.plot(First_half)
plt.subplot(2,1,2)
plt.plot(Sec_half)
plt.show()


# In[5]:


def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b)/la.norm(a)
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    return rmse, mae, 1-F_norm, r2


# In[6]:


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    data1 = np.mat(data)
    train_size = int(time_len * rate)
    train_data = data1[0:train_size]
    test_data = data1[train_size:time_len]
    
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0 : seq_len])
        trainY.append(a[seq_len : seq_len + pre_len])
    for i in range(len(test_data) - seq_len -pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0 : seq_len])
        testY.append(b[seq_len : seq_len + pre_len])
    return trainX, trainY, testX, testY


# In[10]:


time_len = df_removez.shape[0]
num_nodes = df_removez.shape[1]
train_rate = 0.7
seq_len = 24
pre_len = 1
trainX,trainY,testX,testY = preprocess_data(df_removez, time_len, train_rate, seq_len, pre_len)
method = "HA"


# In[11]:


if method == 'HA':
    result = []
    for i in range(len(testX)):
        a = testX[i]
        a1 = np.mean(a, axis=0) 
        result.append(a1)
    result1 = np.array(result)
    result1 = np.reshape(result1, [-1,num_nodes])
    testY1 = np.array(testY)
    testY1 = np.reshape(testY1, [-1,num_nodes])
    rmse, mae, accuracy,r2 = evaluation(testY1, result1)  
    print('HA_rmse:%r'%rmse,
          'HA_mae:%r'%mae,
          'HA_acc:%r'%accuracy,
          'HA_r2:%r'%r2)


# In[9]:



############ SVR #############
if method == 'SVR':  
    total_rmse, total_mae, total_acc, result = [], [],[],[]
    for i in range(num_nodes):
        data1 = np.mat(df_removez)
        a = data1[:,i]
        a_X, a_Y, t_X, t_Y = preprocess_data(a, time_len, train_rate, seq_len, pre_len)
        a_X = np.array(a_X)
        a_X = np.reshape(a_X,[-1, seq_len])
        a_Y = np.array(a_Y)
        a_Y = np.reshape(a_Y,[-1, pre_len])
        a_Y = np.mean(a_Y, axis=1)
        t_X = np.array(t_X)
        t_X = np.reshape(t_X,[-1, seq_len])
        t_Y = np.array(t_Y)
        t_Y = np.reshape(t_Y,[-1, pre_len])    
       
        svr_model=SVR(kernel='linear')
        svr_model.fit(a_X, a_Y)
        pre = svr_model.predict(t_X)
        pre = np.array(np.transpose(np.mat(pre)))
        pre = pre.repeat(pre_len ,axis=1)
        result.append(pre)
    result1 = np.array(result)
    result1 = np.reshape(result1, [num_nodes,-1])
    result1 = np.transpose(result1)
    testY1 = np.array(testY)


    testY1 = np.reshape(testY1, [-1,num_nodes])
    total = np.mat(total_acc)
    total[total<0] = 0
    rmse1, mae1, acc1,r2,var = evaluation(testY1, result1)
    print('SVR_rmse:%r'%rmse1,
          'SVR_mae:%r'%mae1,
          'SVR_acc:%r'%acc1,
          'SVR_r2:%r'%r2,
          'SVR_var:%r'%var)


# In[ ]:




