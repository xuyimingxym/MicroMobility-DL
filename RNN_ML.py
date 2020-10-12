#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 22:49:27 2020

@author: m.paliwal
"""
#Load the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers





#Loading the Sample-Dataset
df_jun = pd.read_csv('More_data/GridCount_Jun_400m.csv')

df_jun = df_jun.rename(columns={x:y for x,y in zip(df_jun.columns[1:],range(0,len(df_jun.columns)))})


df_jul = pd.read_csv('More_data/GridCount_Jul_400m.csv')

df_aug = pd.read_csv('More_data/GridCount_Aug_400m.csv')

df_Sep = pd.read_csv('More_data/GridCount_Sep_400m.csv')

df_oct = pd.read_csv('More_data/GridCount_Oct_400m.csv')

df_Nov = pd.read_csv('More_data/GridCount_Nov_400m.csv')

df_Dec = pd.read_csv('More_data/GridCount_Dec_400m.csv')


df = pd.concat([df_jun, df_jul, df_aug, df_Sep, df_oct, df_Nov, df_Dec], axis=1,ignore_index=True)

df = df.drop("pos_code",axis=1)                  # dropping the position code column


df_t = df[(df.T!= 0).any()]
cln_dataset = df_t.T   
cln_dataset.columns = cln_dataset.columns.map(str)               # removing the grids with zero demand values 

cln_dataset.describe().transpose()                  # looking at statisitics of the dataset


First_half=cln_dataset.iloc[:,:612]
Sec_half=cln_dataset.iloc[:,612:]#df_removez.columns.get_loc(41):df_removez.columns.get_loc(550)]
First_half.index = np.array(range(cln_dataset.shape[0]))
Sec_half.index = np.array(range(cln_dataset.shape[0]))
plt.figure(figsize=(25,20))
plt.subplot(2,1,1)
plt.plot(First_half)
plt.subplot(2,1,2)
plt.plot(Sec_half)
plt.show()





#split the data 
column_indices = {name:i for i,name in enumerate(cln_dataset.columns)}

n=len(cln_dataset)
train_df = cln_dataset[0:int(n*0.7)]
val_df = cln_dataset[int(n*0.7):int(n*0.9)]
test_df = cln_dataset[int(n*0.9):]

num_features = cln_dataset.shape[1]

num_nodes = cln_dataset.shape[1]

# Normalizing the data 
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean)/train_std
val_df = (val_df-train_mean)/ train_std
test_df = (test_df-train_mean)/train_std




# window generator 
class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                train_df =train_df, val_df=val_df, test_df=test_df,
                label_columns=None ):
            
        #store the raw data 
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        #work out on the label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name:i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in 
                               enumerate(train_df.columns)}
                               
        # Work out the window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        
        self.total_window_size = input_width +shift
        
        self.input_slice = slice(0,input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
    def __repr__(self):
        return '\n'.join([
            f'Total window size:{self.total_window_size}',
            f'Input indices:{self.input_indices}',
            f'Label indices:{self.label_indices}',
            f'Label column name(s):{self.label_columns}'])
    
    
w1=WindowGenerator(input_width=24, label_width=1, shift=1,
                   label_columns=train_df.columns[0])

w2=WindowGenerator(input_width=6, label_width=1, shift=1,
                   label_columns=train_df.columns[0])
        
        
    









    
#splitting the data into windows and labels
def split_window(self,features):
    inputs = features[:,self.input_slice,:]
    labels = features[:, self.labels_slice,:]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:,:,self.column_indices[name]] for name in self.label_columns],
            axis=-1)
        
    
    inputs.set_shape([None,self.input_width,None])
    labels.set_shape([None, self.label_width,None])
    
    return inputs, labels

WindowGenerator.split_window=split_window
        
# checking the window size     
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size]),
                           ])


example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'labels shape: {example_labels.shape}')     
        
w2.example = example_inputs,example_labels
      










  
        
def plot(self, model=None, plot_col='286', max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(3, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [h]')

WindowGenerator.plot = plot        
        
        
        
        
      
		
	  


def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset  








@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example        
        
w2.train.element_spec        
        
        
        
for example_inputs, example_labels in w2.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')  




class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]











OUT_STEPS = 24
multi_window = WindowGenerator(input_width=24,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)

multi_window.plot()
multi_window
        
for example_inputs, example_labels in multi_window.train.take(2):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')
  
  
  
class MultiStepLastBaseline(tf.keras.Model):
  def call(self, inputs):
    return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance = {}
multi_performance = {}

multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
multi_performance['Last'] = last_baseline.evaluate(multi_window.test, verbose=1)
multi_window.plot(last_baseline)


class RepeatBaseline(tf.keras.Model):
  def call(self, inputs):
    return inputs

repeat_baseline = RepeatBaseline()
repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test, verbose=0)
multi_window.plot(repeat_baseline)








MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history


import IPython
import IPython.display







multi_linear_model = tf.keras.Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_linear_model, multi_window)

IPython.display.clear_output()
multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_linear_model)
