# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:03:30 2019

@author: Shivanjali Khare
"""


import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

Column_Names = ['user', 'activity','timestamp','x-axis','y-axis','z-axis']
Labels = ['Downstairs','Jogging','Sitting','Standing','Upstairs','Walking']
Data_path = 'data/WISDM_ar_v1.1_raw.txt'

time_step = 100
num_class = 6
num_features = 3
time_size_segment = 180

#load data
data = pd.read_csv(Data_path, header=None, names=Column_Names)
data['z-axis'].replace({';':''}, regex=True, inplace=True)
data=data.dropna()
#print(data.head())

#show graph for any activity
data[data['activity'] == 'Standing'][['x-axis']][:50].plot(subplots=True,figsize=(10,8),title='Standing')
data[data['activity'] == 'Jogging'][['x-axis']][:50].plot(subplots=True,figsize=(10,8),title='Jogging')
plt.xlabel('Timestep')
plt.ylabel('x acc (dg)')

#activity graph
activity_type = data['activity'].value_counts().plot(kind='bar', title='frequency of activity')
plt.show()

#data_after_window_segmentation 
data_after_window_segmentation = []
labels = []

for i in range(0,len(data) - time_size_segment,time_step):
    x = data['x-axis'].values[i:i + time_size_segment]
    y = data['y-axis'].values[i:i + time_size_segment]
    z = data['z-axis'].values[i:i + time_size_segment]
    data_after_window_segmentation.append([x,y,z])
    
    label = stats.mode(data['activity'][i: i+time_size_segment])[0][0]
    labels.append(label)
    
#convert the new data to numpy
data_after_window_segmentation = np.asarray(data_after_window_segmentation, dtype=np.float32).transpose(0,2,1)   
labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
print("data_after_window_segmentation shape:",data_after_window_segmentation.shape)
print("labels shape",labels.shape) 