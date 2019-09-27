# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:03:30 2019

@author: Shivanjali Khare
"""


import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import iqr

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
#data[data['activity'] == 'Standing'][['x-axis']][:50].plot(subplots=True,figsize=(10,8),title='Standing')
#data[data['activity'] == 'Jogging'][['x-axis']][:50].plot(subplots=True,figsize=(10,8),title='Jogging')
#plt.xlabel('Timestep')
#plt.ylabel('x acc (dg)')

#activity graph
#activity_type = data['activity'].value_counts().plot(kind='bar', title='frequency of activity')
#plt.show()

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



print(data_after_window_segmentation[:1,:1])
print(data_after_window_segmentation[0][:1,:1])
print(len(data_after_window_segmentation))

total_acc = []
for i in range(0, len(data_after_window_segmentation)):
    a = np.sqrt(np.sum(np.square(data_after_window_segmentation[:i,:i])))
        
    total_acc.append(a)
    
#print(len(total_acc))
#print(total_acc)
        
x = data_after_window_segmentation[:,:,0]
y = data_after_window_segmentation[:,:,1]
z = data_after_window_segmentation[:,:,2]


x_mean = np.mean(x)
y_mean = np.mean(y)
z_mean = np.mean(z)
total_acc_mean = np.mean(total_acc)
print('x_mean',x_mean)
#print('---------------------------------------------')
print('y_mean',y_mean)
#print('---------------------------------------------')
print('z_mean',z_mean)
#print('---------------------------------------------')
print('total_acc_mean',total_acc_mean)

x_var = np.var(x, dtype=np.float32)
y_var = np.var(y, dtype=np.float32)
z_var = np.var(z, dtype=np.float32)
total_acc_var = np.var(total_acc, dtype=np.float32)
print('x_var',x_var)
#print('---------------------------------------------')
print('y_var',y_var)
#print('---------------------------------------------')
print('z_var',z_var)
#print('---------------------------------------------')
print('total_acc_var',total_acc_var)
x_covariance = np.cov(x)
y_covariance = np.cov(y)
y_covariance = np.cov(y)
x_skew = skew(x)
y_skew = skew(y)
z_skew = skew(z)
total_acc_skew = skew(total_acc)
#print('skew',x_skew)
kurtosis
x_kurtosis = kurtosis(x)
y_kurtosis = kurtosis(y)
z_kurtosis = kurtosis(z)
total_acc_kurtosis = kurtosis(total_acc)
#print('kurtosis',len(x_kurtosis))
x_std = np.std(x)
y_std = np.std(y)
z_std = np.std(z)
total_acc_std = np.std(total_acc)
#print('x_std',x_std)


x_rms = np.sqrt(np.mean(x**2))
y_rms = np.sqrt(np.mean(y**2))
z_rms = np.sqrt(np.mean(z**2))

positions = np.cumsum(total_acc, axis=0)
t_sq = positions**2
t_mean_sq = np.mean(t_sq, axis= 0)
total_acc_rms = np.sqrt(t_mean_sq)

#print('x_rms',x_rms)
#print('y_rms',y_rms)
#print('z_rms',z_rms)
#print('total_acc_rms',total_acc_rms)

x_iqr = iqr(x)
y_iqr = iqr(y)
z_iqr = iqr(z)
total_acc_iqr = iqr(total_acc)
print('x_iqr',x_iqr)
print('y_iqr',y_iqr)
print('z_iqr',z_iqr)

print('total_acc_iqr',total_acc_iqr)

x_min = np.amin(x)
y_min = np.amin(y)
z_min = np.amin(z)
total_acc_min = np.amin(total_acc)
x_max = np.amax(x)
y_max = np.amax(y)
z_max = np.amax(z)
total_acc_max = np.amax(total_acc)

print("x_min  ",x_min,"       y_min   ",y_min,"     z_min  ",z_min, "  total_acc_min  ", total_acc_min)
print("x_max  ",x_max,"       y_max   ",y_max,"     z_max  ",z_max, "  total_acc_max  ", total_acc_max)


#find mean trend
def Mean(a):
    cumsum, moving_aves = [0], []
    moving_aves_w = []
    N = 90
    for i, ii in enumerate(a, 1):
        cumsum.append(cumsum[i-1] + ii)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_ave += moving_ave
            moving_aves.append(moving_ave)
            moving_ave_w = moving_ave - np.mean(a)
            moving_aves_w.append(moving_ave_w)
            return moving_aves, moving_aves_w

print('x_meanTrend',Mean(x)[0])
print('y_meanTrend',Mean(y)[0])
print('z_meanTrend',Mean(z)[0])
print('total_acc_meanTrend',Mean(total_acc)[0])
print('x_WindowMeanDifference',Mean(x)[1])
print('y_WindowMeanDifference',Mean(x)[1])
print('z_WindowMeanDifference',Mean(x)[1])
print('total_acc_WindowMeanDifference',Mean(total_acc)[1])






#print(data_after_window_segmentation[0,0])
#print(data_after_window_segmentation[:,:,0])
    

    


    
