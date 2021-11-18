#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 18:57:56 2021

@author: Arturo Collado Rosell
"""


import sys
import os
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'machine_learning_scripts/')
import RadarNet 

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession 
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


#number of categories, from the some_params_to_train.npy is extracted this information
try:
    meta_params = np.load('some_params_to_train.npy')
    N_vel = int(meta_params[0])
    N_s_w = int(meta_params[1])
    N_csr = int(meta_params[2])
    operation_mode = str(meta_params[3])
    training_data = np.load('training_data.npy')
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

#######################Diferent models to train################################
#for conv key [x,y] means x kernels of lenght y, for dense key each number represent the number of neurons in a dense layer 
dict_vel_layers_1 = {'conv':[[3,20]], 'dense':[80, 60] }
dict_width_layers_1 = {'conv':[[3,20]], 'dense':[80, 40] }
dict_csr_layers_1 = {'conv':[[3,20]], 'dense':[80, 50] }

dict_vel_layers_2 = {'conv':[[6,20]], 'dense':[80, 60] }
dict_width_layers_2 = {'conv':[[6,20]], 'dense':[80, 40] }
dict_csr_layers_2 = {'conv':[[6,20]], 'dense':[80, 50] }

dict_vel_layers_3 = {'conv':[[3,20]], 'dense':[150, 100] }
dict_width_layers_3 = {'conv':[[3,20]], 'dense':[150, 80] }
dict_csr_layers_3 = {'conv':[[3,20]], 'dense':[150, 90] }

dict_vel_layers_4 = {'conv':[[6,20]], 'dense':[150, 100] }
dict_width_layers_4 = {'conv':[[6,20]], 'dense':[150, 80] }
dict_csr_layers_4 = {'conv':[[6,20]], 'dense':[150, 90] }

dict_vel_layers_5 = {'conv':[[3,20], [6,20]], 'dense':[80, 60] }
dict_width_layers_5 = {'conv':[[3,20], [6,20]], 'dense':[80, 40] }
dict_csr_layers_5 = {'conv':[[3,20], [6,20]], 'dense':[80, 50] }

dict_vel_layers_6 = {'conv':[[3,20], [6,20]], 'dense':[150, 100] }
dict_width_layers_6 = {'conv':[[3,20], [6,20]], 'dense':[150, 80] }
dict_csr_layers_6 = {'conv':[[3,20], [6,20]], 'dense':[150, 90] }

dict_vel_layers_7 = {'conv':[[5,5], [5,5]], 'dense':[150, 100] }
dict_width_layers_7 = {'conv':[[5,5], [5,5]], 'dense':[150, 80] }
dict_csr_layers_7 = {'conv':[[5,5], [5,5]], 'dense':[150, 90] }


dict_vel_layers_8 = {'conv':[[3,20], [6,20], [9,20]], 'dense':[80, 60] }
dict_width_layers_8 = {'conv':[[3,20], [6,20], [9,20]], 'dense':[80, 40] }
dict_csr_layers_8 = {'conv':[[3,20], [6,20], [9,20]], 'dense':[80, 50] }

dict_vel_layers_9 = {'conv':[[3,20], [6,20], [9,20]], 'dense':[150, 100] }
dict_width_layers_9 = {'conv':[[3,20], [6,20], [9,20]], 'dense':[150, 80] }
dict_csr_layers_9 = {'conv':[[3,20], [6,20], [9,20]], 'dense':[150, 90] }

dict_vel_layers_10 = {'conv':[[5,5], [5,5],[5,5]], 'dense':[80, 60] }
dict_width_layers_10 = {'conv':[[5,5], [5,5], [5,5]], 'dense':[80, 40] }
dict_csr_layers_10 = {'conv':[[5,5], [5,5], [5,5]], 'dense':[80, 50] }

dict_vel_layers_11 = {'conv':[[5,5], [5,5],[5,5]], 'dense':[100, 80, 60] }
dict_width_layers_11 = {'conv':[[5,5], [5,5], [5,5]], 'dense':[100, 80, 40] }
dict_csr_layers_11 = {'conv':[[5,5], [5,5], [5,5]], 'dense':[100, 80, 50] }

dict_vel_layers_12 = {'conv':[[5,5], [5,5],[5,5], [5,5], [5,5]], 'dense':[200, 100, 80, 60] }
dict_width_layers_12 = {'conv':[[5,5], [5,5], [5,5], [5,5], [5,5]], 'dense':[200, 100, 80, 40] }
dict_csr_layers_12 = {'conv':[[5,5], [5,5], [5,5], [5,5], [5,5]], 'dense':[200, 100, 80, 50] }

velocity_branch_NNs = [dict_vel_layers_1, dict_vel_layers_2, dict_vel_layers_3, dict_vel_layers_4, dict_vel_layers_5, dict_vel_layers_6, dict_vel_layers_7, dict_vel_layers_8, dict_vel_layers_9, dict_vel_layers_10, dict_vel_layers_11, dict_vel_layers_12]
sw_branch_NNs = [dict_width_layers_1, dict_width_layers_2, dict_width_layers_3, dict_width_layers_4, dict_width_layers_5, dict_width_layers_6, dict_width_layers_7, dict_width_layers_8, dict_width_layers_9, dict_width_layers_10, dict_width_layers_11, dict_width_layers_12]
csr_branch_NNs = [dict_csr_layers_1, dict_csr_layers_2, dict_csr_layers_3, dict_csr_layers_4, dict_csr_layers_5, dict_csr_layers_6, dict_csr_layers_7, dict_csr_layers_8, dict_csr_layers_9, dict_csr_layers_10, dict_csr_layers_11, dict_csr_layers_12]
#############
EPOCHS = 150
BS = 512
lr = 1e-4

for i, vel_dict, sw_dict, csr_dict in zip(np.arange(1,13,1) ,velocity_branch_NNs, sw_branch_NNs, csr_branch_NNs):
    
    dirName =f'structure_number_{i}' + device[1:4] + 'entrenamiento'+ '_' +str(M)+'_'  +'/'
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")
    model = RadarNet.create_convolutional_network(M, N_vel, N_s_w, N_csr, vel_dict, sw_dict, csr_dict)
    
    H = RadarNet.model_compile_and_train(device,model, X_train, y_train_cat_vel, y_train_cat_s_w, y_train_cat_csr, X_test, y_test_cat_vel, y_test_cat_s_w, y_test_cat_csr, dirName, EPOCHS, BS, lr)
    RadarNet.plot_training(H, dirName)
    