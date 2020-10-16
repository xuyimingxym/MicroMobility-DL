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


# # Generate Grid

# In[2]:


def covert_coordinate_from_4326_to_DC(lat,lon):
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


all_boundary=pd.read_csv('../Scooter_study/Washington_DC_26985Boundary.csv')


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


# In[6]:


pos_code_index=pd.DataFrame()
pos_code_index['pos_code']=points['x']*10**6+points['y']
pos_code_index.index=pos_code_index['pos_code']
print(pos_code_index.shape)


# # Neighborhood

# In[11]:


d=400 #grid size
nb=pd.DataFrame(index=pos_code_index.pos_code,columns=pos_code_index.pos_code)
for i in nb.columns:
    for j in nb.index:
        xi=np.round(i/10**6)
        yi=i%10**6
        xj=np.round(j/10**6)
        yj=j%10**6
        dx=abs(xi-xj)
        dy=abs(yi-yj)
        if (dx>d) | (dy>d):
            nb[i][j]=0
        if (dx<=d) & (dy<=d):
            nb[i][j]=1
        if (dx+dy)==0:
            nb[i][j]=0
            
nb.to_csv('neighborhood.csv')


# # Point of Interest

# In[46]:


poi_data=pd.read_csv('POI_Count.csv')
poi_data.index=pos_code_index.index
poi=pd.DataFrame(index=pos_code_index.index,columns=pos_code_index.index)
poi.fillna(1,inplace=True)
poi[:][poi_data['Count']==0]=0
poi.loc[:,poi_data['Count']==0]=0
poi.to_csv('POI_Similarity.csv')


# # Connectivity

# In[37]:


conn_data=pd.read_csv('Street_Count.csv') # Street_Count.csv counts how many streets in a cell
conn_data.index=pos_code_index.index
conn=pd.DataFrame(index=pos_code_index.index,columns=pos_code_index.index)
conn.fillna(1,inplace=True)
conn[:][conn_data['Count']==0]=0
conn.loc[:,conn_data['Count']==0]=0


# In[59]:


neighborhood=pd.read_csv('neighborhood.csv')
neighborhood.index=neighborhood['pos_code']
neighborhood.drop('pos_code',axis=1,inplace=True)


# In[55]:


conn_output=conn-neighborhood
conn_output.replace(-1,0,inplace=True)


# In[60]:


conn_output.to_csv('Street_Connectivity.csv')

