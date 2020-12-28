#!/usr/bin/env python
# coding: utf-8

# In[2]:


# input_data
import numpy as np
import pandas as pd
import pickle as pkl



def load_dc_data(dataset):
    dc_adj1 = pd.read_csv('C:/YimingXu/Micromobility_DL/data/adjacency_selected.csv')
    adj1 = np.mat(dc_adj1)
    dc_adj2 = pd.read_csv('C:/YimingXu/Micromobility_DL/data/accessibility_selected.csv')
    adj2 = np.mat(dc_adj2)
    dc_adj3 = pd.read_csv('C:/YimingXu/Micromobility_DL/data/landuse_selected.csv')
    adj3 = np.mat(dc_adj3)
    dc_adj4 = pd.read_csv('C:/YimingXu/Micromobility_DL/data/demographic_selected.csv')
    adj4 = np.mat(dc_adj4)
    dc_dm = pd.read_pickle('C:/YimingXu/Micromobility_DL/data/Input_Selected_Zones.pkl')
    return dc_dm, adj1, adj2, adj3, adj4


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]
    
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0 : seq_len])
        trainY.append(a[seq_len : seq_len + pre_len])
    for i in range(len(test_data) - seq_len -pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0 : seq_len])
        testY.append(b[seq_len : seq_len + pre_len])
      
    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    return trainX1, trainY1, testX1, testY1


# In[3]:


# utils
import tensorflow as tf
import scipy.sparse as sp
import numpy as np

def normalized_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    normalized_adj = normalized_adj.astype(np.float32)
    return normalized_adj
    
def sparse_to_tuple(mx):
    mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    L = tf.SparseTensor(coords, mx.data, mx.shape)
    return tf.sparse.reorder(L) 
    
def calculate_laplacian(adj, lambda_max=1):  
    adj = normalized_adj(adj + sp.eye(adj.shape[0]))
    adj = sp.csr_matrix(adj)
    adj = adj.astype(np.float32)
    return sparse_to_tuple(adj)
    
def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                            maxval=init_range, dtype=tf.float32)

    return tf.Variable(initial,name=name)  


# In[4]:


# TGCN Cell

from tensorflow.compat.v1.nn.rnn_cell import RNNCell

class tgcnCell(RNNCell):
    """Temporal Graph Convolutional Network """

    def call(self, inputs, **kwargs):
        pass

    def __init__(self, num_units, adj, num_nodes, input_size=None,
                 act=tf.nn.tanh, reuse=None):

        super(tgcnCell, self).__init__(_reuse=reuse)
        self._act = act
        self._nodes = num_nodes
        self._units = num_units
        self._adj = []
        self._adj.append(calculate_laplacian(adj))


    @property
    def state_size(self):
        return self._nodes * self._units

    @property
    def output_size(self):
        return self._units

    def __call__(self, inputs, state, scope=None):

        with tf.compat.v1.variable_scope(scope or "tgcn",reuse=tf.compat.v1.AUTO_REUSE):
            with tf.compat.v1.variable_scope("gates",reuse=tf.compat.v1.AUTO_REUSE):  
                value = tf.nn.sigmoid(
                    self._gc(inputs, state, 2 * self._units, bias=1.0, scope=scope))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
            with tf.compat.v1.variable_scope("candidate",reuse=tf.compat.v1.AUTO_REUSE):
                r_state = r * state
                c = self._act(self._gc(inputs, r_state, self._units, scope=scope))
            new_h = u * state + (1 - u) * c
        return new_h, new_h


    def _gc(self, inputs, state, output_size, bias=0.0, scope=None):
        ## inputs:(-1,num_nodes)
        inputs = tf.expand_dims(inputs, 2)
        ## state:(batch,num_node,gru_units)
        state = tf.reshape(state, (-1, self._nodes, self._units))
        ## concat
        x_s = tf.concat([inputs, state], axis=2)
        input_size = x_s.get_shape()[2]
        ## (num_node,input_size,-1)
        x0 = tf.transpose(x_s, perm=[1, 2, 0])  
        x0 = tf.reshape(x0, shape=[self._nodes, -1])
        
        scope = tf.compat.v1.get_variable_scope()
        with tf.compat.v1.variable_scope(scope):
            for m in self._adj:
                x1 = tf.sparse.sparse_dense_matmul(m, x0)
#                print(x1)
            x = tf.reshape(x1, shape=[self._nodes, input_size,-1])
            x = tf.transpose(x,perm=[2,0,1])
            x = tf.reshape(x, shape=[-1, input_size])
            weights = tf.compat.v1.get_variable(
                'weights', [input_size, output_size], initializer=tf.keras.initializers.glorot_normal)
            x = tf.matmul(x, weights)  # (batch_size * self._nodes, output_size)
            biases = tf.compat.v1.get_variable(
                "biases", [output_size], initializer=tf.constant_initializer(bias))
            x = tf.nn.bias_add(x, biases)
            x = tf.reshape(x, shape=[-1, self._nodes, output_size])
            x = tf.reshape(x, shape=[-1, self._nodes * output_size])
        return x


# In[5]:


import pickle as pkl
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la

from sklearn.metrics import mean_squared_error,mean_absolute_error
import time

time_start = time.time()
###### Settings ######
# flags = tf.compat.v1.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
# flags.DEFINE_integer('training_epoch', 1, 'Number of epochs to train.')
# flags.DEFINE_integer('gru_units', 64, 'hidden units of gru.')
# flags.DEFINE_integer('seq_len',12 , '  time length of inputs.')
# flags.DEFINE_integer('pre_len', 3, 'time length of prediction.')
# flags.DEFINE_float('train_rate', 0.8, 'rate of training set.')
# flags.DEFINE_integer('batch_size', 32, 'batch size.')
# flags.DEFINE_string('dataset', 'los', 'sz or los.')
# flags.DEFINE_string('model_name', 'tgcn', 'tgcn')
model_name = 'tgcn'
data_name = 'dc'
train_rate =  0.8
seq_len = 24
output_dim = pre_len = 3
batch_size = 32
lr = 0.001
training_epoch = 1
gru_units = 64


# In[6]:


###### load data ######
if data_name == 'dc':
    data, adj1, adj2, adj3, adj4 = load_dc_data('dc')

time_len = data.shape[0]
num_nodes = data.shape[1]
data1 =np.mat(data,dtype=np.float32)


# In[7]:


#### normalization
# max_value = np.max(data1)
# data1  = data1/max_value
max_value=1
mean_value=np.mean(data1)
std_value=np.std(data1)
data1=(data1-mean_value)/std_value
trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)

totalbatch = int(trainX.shape[0]/batch_size)
training_data_count = len(trainX)


# In[8]:


def process_output(otp):
    m = []
    for i in otp:
        o = tf.reshape(i,shape=[-1,num_nodes,gru_units])
        o = tf.reshape(o,shape=[-1,gru_units])
        m.append(o)
    return m


# In[9]:


# TGCN
from tensorflow import keras
def TGCN(_X, _weights, _biases):
    ###
    # multi-GCN-GRU
    cell_1 = tgcnCell(gru_units, adj1, num_nodes=num_nodes)
    cell_2 = tgcnCell(gru_units, adj2, num_nodes=num_nodes)
    cell_3 = tgcnCell(gru_units, adj3, num_nodes=num_nodes)
    cell_4 = tgcnCell(gru_units, adj4, num_nodes=num_nodes)
    
    cell_11 = tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    cell_22 = tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell_2], state_is_tuple=True)
    cell_33 = tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell_3], state_is_tuple=True)
    cell_44 = tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell_4], state_is_tuple=True)
    
    _X = tf.unstack(_X, axis=1)
    
    outputs_1, states_1 = tf.compat.v1.nn.static_rnn(cell_11, _X, dtype=tf.float32)
    outputs_2, states_2 = tf.compat.v1.nn.static_rnn(cell_22, _X, dtype=tf.float32)
    outputs_3, states_3 = tf.compat.v1.nn.static_rnn(cell_33, _X, dtype=tf.float32)
    outputs_4, states_4 = tf.compat.v1.nn.static_rnn(cell_44, _X, dtype=tf.float32)
    
    m_1 = process_output(outputs_1)
    m_2 = process_output(outputs_2)
    m_3 = process_output(outputs_3)
    m_4 = process_output(outputs_4)
    
    last_output_1 = m_1[-1]
    last_output_2 = m_2[-1]
    last_output_3 = m_3[-1]
    last_output_4 = m_4[-1]
    
    dense_input = tf.concat([last_output_1, last_output_2, last_output_3, last_output_4], 1)
    
    # Dense
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(64))
    last_output = model(dense_input)
    
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output,shape=[-1,num_nodes,pre_len])
    output = tf.transpose(output, perm=[0,2,1])
    output = tf.reshape(output, shape=[-1,num_nodes])
    
    return output, m_1 , states_1
        


# In[10]:


###### placeholders ######
tf.compat.v1.disable_eager_execution()
inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
labels = tf.compat.v1.placeholder(tf.float32, shape=[None, pre_len, num_nodes])


# In[11]:


# Graph weights
weights = {
    'out': tf.Variable(tf.compat.v1.random_normal([gru_units, pre_len], mean=1.0), name='weight_o')}
biases = {
    'out': tf.Variable(tf.compat.v1.random_normal([pre_len]),name='bias_o')}

if model_name == 'tgcn':
    pred,ttts,ttto = TGCN(inputs, weights, biases)

y_pred = pred


# In[12]:


###### optimizer ######
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.compat.v1.trainable_variables())
label = tf.reshape(labels, [-1,num_nodes])
##loss
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred-label) + Lreg)
##rmse
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred-label)))
optimizer = tf.compat.v1.train.AdamOptimizer(lr).minimize(loss)


# In[13]:


###### Initialize session ######
variables = tf.compat.v1.global_variables()
saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())  
#sess = tf.Session()
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
sess.run(tf.compat.v1.global_variables_initializer())

out = 'out/%s'%(model_name)
#out = 'out/%s_%s'%(model_name,'perturbation')
path1 = '%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r'%(model_name,data_name,lr,batch_size,gru_units,seq_len,pre_len,training_epoch)
path = os.path.join(out,path1)
if not os.path.exists(path):
    os.makedirs(path)


# In[15]:


###### evaluation ######
def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b,'fro')/la.norm(a,'fro')
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a-b))/np.var(a)
    return rmse, mae, 1-F_norm, r2, var
 
   
x_axe,batch_loss,batch_rmse,batch_pred = [], [], [], []
test_loss,test_rmse,test_mae,test_acc,test_r2,test_var,test_pred = [],[],[],[],[],[],[]

training_epoch = 20
for epoch in range(training_epoch):
    for m in range(totalbatch):
        mini_batch = trainX[m * batch_size : (m+1) * batch_size]
        mini_label = trainY[m * batch_size : (m+1) * batch_size]
        _, loss1, rmse1, train_output = sess.run([optimizer, loss, error, y_pred],
                                                 feed_dict = {inputs:mini_batch, labels:mini_label})
        batch_loss.append(loss1)
        batch_rmse.append(rmse1 * max_value)

     # Test completely at every epoch
    loss2, rmse2, test_output = sess.run([loss, error, y_pred],
                                         feed_dict = {inputs:testX, labels:testY})
    test_label = np.reshape(testY,[-1,num_nodes])
    rmse, mae, acc, r2_score, var_score = evaluation(test_label, test_output)
    test_label1 = test_label * max_value
    test_output1 = test_output * max_value
    test_loss.append(loss2)
    test_rmse.append(rmse * max_value)
    test_mae.append(mae * max_value)
    test_acc.append(acc)
    test_r2.append(r2_score)
    test_var.append(var_score)
    test_pred.append(test_output1)
    
    print('Iter:{}'.format(epoch),
          'train_rmse:{:.4}'.format(batch_rmse[-1]),
          'test_loss:{:.4}'.format(loss2),
          'test_rmse:{:.4}'.format(rmse),
          'test_mae:{:.4}'.format(mae))
    
    if (epoch % 500 == 0):        
        saver.save(sess, path+'/model_100/TGCN_pre_%r'%epoch, global_step = epoch)
        
time_end = time.time()
print(time_end-time_start,'s')


# In[ ]:





# In[ ]:





# In[120]:


############## visualization ###############
b = int(len(batch_rmse)/totalbatch)
batch_rmse1 = [i for i in batch_rmse]
train_rmse = [(sum(batch_rmse1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]
batch_loss1 = [i for i in batch_loss]
train_loss = [(sum(batch_loss1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]

index = test_rmse.index(np.min(test_rmse))
test_result = test_pred[index]
var = pd.DataFrame(test_result)
# var.to_csv(path+'/test_result.csv',index = False,header = False)
#plot_result(test_result,test_label1,path)
#plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,path)

print('min_rmse:%r'%(np.min(test_rmse)),
      'min_mae:%r'%(test_mae[index]),
      'max_acc:%r'%(test_acc[index]),
      'r2:%r'%(test_r2[index]),
      'var:%r'%test_var[index])

