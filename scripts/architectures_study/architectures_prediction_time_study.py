#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 15:52:11 2021

@author: Arturo Collado Rosell
"""

import sys
import os
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')
import RadarNet 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession 
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


try:
    meta_params = np.load('../training_data/some_params_to_train.npy')
    N_vel = int(meta_params[0])
    N_s_w = int(meta_params[1])
    N_csr = int(meta_params[2])
    operation_mode = str(meta_params[3])
    training_data = np.load('../training_data/training_data.npy')
    M = training_data.shape[1]-3
    X = training_data[:,:M].copy()
    y = training_data[:,M:].copy()
    del training_data
except Exception as e:
    print(e)     
    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 29)
#converting the labels to categorical 
y_train_cat_csr = tf.keras.utils.to_categorical(y_train[:,0], N_csr)
y_train_cat_vel = tf.keras.utils.to_categorical(y_train[:,1], N_vel)
y_train_cat_s_w = tf.keras.utils.to_categorical(y_train[:,2], N_s_w)

y_test_cat_csr = tf.keras.utils.to_categorical(y_test[:,0], N_csr)
y_test_cat_vel = tf.keras.utils.to_categorical(y_test[:,1], N_vel)
y_test_cat_s_w = tf.keras.utils.to_categorical(y_test[:,2], N_s_w)


# device = '/CPU:0'
device = '/GPU:0'
architecture = ['0','1','2','3','4','5','6','7','8','9','10','11']


L = 10
elapsed_time = np.zeros((len(architecture),L))  
elapsed_time_mean = np.zeros((len(architecture),1))
elapsed_time_std = np.zeros((len(architecture),1))


dir1 = ''

for t in range(0,L):
   
    for i in range(0,len(architecture)):
                
                directory_to_read_models =  dir1 + 'structure_number_' + str(i+1)  +'CPUentrenamiento_158_/'
                model = tf.keras.models.load_model(directory_to_read_models + 'model.h5')
        
                _,_,_,elapsed_time[i][t] = RadarNet.prediction(model, X_test, device)
    print(t)    
for qq in range(0,elapsed_time.shape[0]):
    elapsed_time_mean[qq] = np.mean(elapsed_time[qq][:])
    elapsed_time_std[qq] = np.std(elapsed_time[qq][:])
            
    
pd.DataFrame(elapsed_time_mean).to_csv('time_to_predict_mean' + device[1:4] + '.csv')
pd.DataFrame(elapsed_time_std).to_csv('time_to_predict_std' + device[1:4] + '.csv')

