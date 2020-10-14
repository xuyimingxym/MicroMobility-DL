#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import json
import numpy as np
from pyproj import Proj, transform
import warnings
import math
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


def covert_coordinate_from_4326_to_DC(lat,lon):
    # covert to meter
    inProj = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:26985')
    lon2, lat2 = transform(inProj, outProj, lon, lat)
    return (lat2, lon2)
def covert_coordinate_from_DC_to_4326(lat,lon):
    outProj = Proj(init='epsg:4326')
    inProj = Proj(init='epsg:26985')
    lon2, lat2 = transform(inProj, outProj, lon, lat)
    return (lat2, lon2)


# In[3]:


all_boundary=pd.read_csv('Washington_DC_26985Boundary.csv') # read the boundary file 


# In[4]:


# All Position Code index
d=400 #grid size 400x400m
points=pd.DataFrame()
all_boundary['0']=round(all_boundary['0']/d)*d 
all_boundary['1']=round(all_boundary['1']/d)*d
x_index=np.sort(all_boundary['0'].unique())
for i in x_index:
    temp=all_boundary.loc[all_boundary['0']==i]
    y_min=min(temp['1'])
    y_max=max(temp['1'])
    if (y_min==y_max) & (i>x_index[1]):
        y_min=y_min0
    y=np.arange(y_min,y_max,d)
    y_min0=y_min
    if len(y)<1:
        continue
    xy=pd.DataFrame()
    xy['y']=y
    xy['x']=i
    points=points.append(xy)

# generate position code based on coordinate of grid
pos_code_index=pd.DataFrame()
pos_code_index['pos_code']=points['x']*10**6+points['y']
pos_code_index.index=pos_code_index['pos_code']
pos_code_index.shape


# In[5]:


plt.figure(figsize=[10,8],dpi=80)
plt.plot(points['y'],points['x'],'.')
plt.show()

