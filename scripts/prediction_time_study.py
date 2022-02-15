#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 01:06:15 2022

@author: acr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data_generator_class import DataGenerator
import tensorflow as tf
import time

optuna_dir = 'models/'
df_optuna = pd.read_csv(optuna_dir+ 'optuna_study.csv')
print(df_optuna.columns)

fig = plt.figure(figsize=(10,6))
plt.plot(df_optuna.number, df_optuna.values_0, '*', label = 'velocity branch')
plt.plot(df_optuna.number, df_optuna.values_1, '<', label = 'width branch')
plt.plot(df_optuna.number, df_optuna.values_2, 'o', label = 'csr branch')
plt.legend()
plt.xlabel('Structures number')
plt.ylabel('Accuracy')
plt.grid()
fig.show()

#saving accuracy values
df_optuna.values_0.to_csv(optuna_dir + 'vel_accu.csv')
df_optuna.values_1.to_csv(optuna_dir + 'width_accu.csv')
df_optuna.values_2.to_csv(optuna_dir + 'csr_accu.csv')

dirName = 'training_data/'

try:
    meta_params = np.load(dirName + 'some_params_to_train.npy')
    N_vel = int(meta_params[0])
    N_s_w = int(meta_params[1])
    N_csr = int(meta_params[2])
    M = int(meta_params[3])
    number_of_batch = int(meta_params[4])
    batch_size = int(meta_params[5])
    operation_mode = str(meta_params[6])

except Exception as e:
    print(e)     
    
validation_IDs = [j for j in range(int(number_of_batch*0.8),number_of_batch)]


#Generators
params = {"dim": M,
          "batch_size": batch_size,
          "n_classes": (N_csr, N_vel, N_s_w)
          }

validation_generator = DataGenerator(validation_IDs, **params)    

L = 10
architectures = 50
device_list = ['/CPU:0', '/GPU:0']

fig = plt.figure(figsize = (5,3))
for device in device_list:
    elapsed_time = np.zeros((architectures,L))
    elapsed_time_mean = np.zeros((architectures,))
    elapsed_time_std = np.zeros((architectures,))
    for i in range(architectures):
        for t in range(0,L):
   

                model = tf.keras.models.load_model(optuna_dir + f'_{i}.h5')
                start = time.time()
                with tf.device(device):
                    model.predict(validation_generator, use_multiprocessing = True, workers=-1)
                                    
                end = time.time()        
                elapsed_time[i][t] = end - start
                print(i,' ', t)    
    for qq in range(0,elapsed_time.shape[0]):
        elapsed_time_mean[qq] = np.mean(elapsed_time[qq][:])
        elapsed_time_std[qq] = np.std(elapsed_time[qq][:])
    pd.DataFrame(elapsed_time_mean).to_csv(optuna_dir +'time_to_predict_mean' + device[1:4] + '.csv')
    pd.DataFrame(elapsed_time_std).to_csv(optuna_dir + 'time_to_predict_std' + device[1:4] + '.csv')    
    plt.plot(elapsed_time_mean,'o', label = 'device[1:4]')

plt.grid()
plt.xlabel('Structure number')
plt.ylabel('time [s]')                
plt.legend()    
fig.show()
# #
struct_and_time = []
for i, value in enumerate(elapsed_time_mean):
    struct_and_time.append([i, value])

#sorting bubble algorithm
for i in range(architectures-1):
    for j in range(architectures - i - 1):
        if struct_and_time[j][1] > struct_and_time[j+1][1]:
            struct_and_time[j], struct_and_time[j+1] = struct_and_time[j+1], struct_and_time[j] 
            
    
    


  
    
    