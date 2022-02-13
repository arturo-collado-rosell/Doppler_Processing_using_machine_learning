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
# df_optuna = pd.read_csv(optuna_dir+ 'optuna_study.csv')
# print(df_optuna.columns)

# fig = plt.figure(figsize=(10,6))
# plt.plot(df_optuna.number, df_optuna.values_0, '*')
# plt.plot(df_optuna.number, df_optuna.values_1, '<')
# plt.plot(df_optuna.number, df_optuna.values_2, 'o')

# plt.grid()
# fig.show()


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
architectures = 30
device = '/CPU:0'
elapsed_time = np.zeros((architectures,L))
elapsed_time_mean = np.zeros((architectures,))
elapsed_time_std = np.zeros((architectures,))
for t in range(0,L):
   
    for i in range(architectures):
                
                
                model = tf.keras.models.load_model(optuna_dir + f'_{i}.h5')
                start = time.time()
                with tf.device(device):
                    model.predict(validation_generator, use_multiprocessing = True, workers=-1)
                                    
                end = time.time()
        
                elapsed_time[i][t] = end - start
    print(t,' ', i)    
for qq in range(0,elapsed_time.shape[0]):
    elapsed_time_mean[qq] = np.mean(elapsed_time[qq][:])
    elapsed_time_std[qq] = np.std(elapsed_time[qq][:])
            
    
pd.DataFrame(elapsed_time_mean).to_csv(optuna_dir +'time_to_predict_mean' + device[1:4] + '.csv')
pd.DataFrame(elapsed_time_std).to_csv(optuna_dir + 'time_to_predict_std' + device[1:4] + '.csv')