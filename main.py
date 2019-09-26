# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:03:30 2019

@author: Shivanjali Khare
"""


import pandas as pd
import matplotlib.pyplot as plt

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