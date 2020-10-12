#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import pyplot
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM

import scipy.sparse as sp
from scipy.sparse import linalg




# In[3]:
#Loading the Sample-Dataset
df_jun = pd.read_csv('More_data/GridCount_Jun_400m.csv')

df_jun = df_jun.rename(columns={x:y for x,y in zip(df_jun.columns[1:],range(0,len(df_jun.columns)))})


df_jul = pd.read_csv('More_data/GridCount_Jul_400m.csv')

df_aug = pd.read_csv('More_data/GridCount_Aug_400m.csv')

df_Sep = pd.read_csv('More_data/GridCount_Sep_400m.csv')

df_oct = pd.read_csv('More_data/GridCount_Oct_400m.csv')

df_Nov = pd.read_csv('More_data/GridCount_Nov_400m.csv')

df_Dec = pd.read_csv('More_data/GridCount_Dec_400m.csv')


df = pd.concat([df_jun, df_jul, df_aug, df_Sep, df_oct, df_Nov, df_Dec], axis=1)

df = df.drop("pos_code",axis=1)                  # dropping the position code column


df_t = df[(df.T!= 0).any()]
cln_dataset = df_t.T   
cln_dataset = cln_dataset.reset_index()
cln_dataset = cln_dataset.drop("index",axis=1)





First_half=cln_dataset.iloc[:,400]
Sec_half=cln_dataset.iloc[:,612:]#df_removez.columns.get_loc(41):df_removez.columns.get_loc(550)]
First_half.index = np.array(range(cln_dataset.shape[0]))
Sec_half.index = np.array(range(cln_dataset.shape[0]))
plt.figure(figsize=(25,20))
plt.subplot(2,1,1)
plt.plot(First_half)
plt.subplot(2,1,2)
plt.plot(Sec_half)
plt.show()



# In[5]:


#transposing the pandas dataframe to bring the time series analysis format
#df_t = df[(df.T!= 0).any()]
#scaled = np.array(df_t.T)
scaled = np.array(cln_dataset)

# In[6]:


time=cln_dataset.shape[0]
time = np.array(range(time))


# In[7]:


#division of data into training, validation and testing set 

split_val_time = 3299
split_test_time = 472

time_train = time[:split_val_time]
x_train = scaled[:split_val_time, :]

time_valid=time[split_val_time : split_test_time+ split_val_time]
x_valid = scaled[split_val_time: split_test_time + split_val_time,:]

time_test = time[split_test_time + split_val_time:]
x_test = scaled[split_test_time + split_val_time:, :]


# In[8]:


# #creating a windowed dataset for the input to the keras deep neural model.
# window_size = 12
# batch_size = 16

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)


# In[9]:


X_multi_train,y_multi_train = multivariate_data(x_train, x_train , 0, 3299, 12, 1, 1,single_step=False )


# In[10]:


x_multi_valid,y_multi_valid = multivariate_data(x_valid, x_valid , 0, 472, 12, 1, 1,single_step=False )


# In[11]:


train_data_single = tf.data.Dataset.from_tensor_slices((X_multi_train, y_multi_train))
#train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_multi_valid, y_multi_valid))
#val_data_single = val_data_single.batch(BATCH_SIZE).repeat()


# In[17]:


# designing a simple LSTM model Network to look at temporal effects on the model

tf.keras.backend.clear_session()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None])),
# Add a LSTM layer with 32 internal units.
#model.add(tf.keras.layers.LSTM(32,activation="tanh",recurrent_activation="sigmoid",return_sequences=True))
model.add(tf.keras.layers.LSTM(16,activation="tanh",recurrent_activation="sigmoid"))

# Add a Dense layer with 1131 units for the number of grids.
model.add(tf.keras.layers.Dense(1024))

model.compile(optimizer='adam', loss='mae', metrics=[tf.keras.metrics.RootMeanSquaredError()])
# model = Sequential()
# model.add(LSTM(32,input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dense(1131))
# model.compile(loss="mse",optimizer="adam")


# In[18]:


model.summary()


# In[19]:


history = model.fit(train_data_single, epochs=5,
                    validation_data=(val_data_single), verbose=1, shuffle=False)


# In[20]:


yhat = model.predict(x_test)
rmse = sqrt(mean_squared_error(x_test, yhat))
print('Test RMSE: %.3f' % rmse)


# In[21]:


results_node = []
for i in range(x_test.shape[1]):
    rmse = sqrt(mean_squared_error(x_test[:,i], yhat[:,i]))
    results_node.append(rmse)


# In[22]:


plt.plot(results_node,color='green')
plt.show()


# In[ ]:




