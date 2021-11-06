#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 18:03:12 2021

@author: Arturo Collado Rosell
"""

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'machine_learning_scripts/')
import RadarNet 

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession 
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


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
# Here you can build your model, every brach separated or the deafult branch networks 

# model = RadarNet.build_all_conv1D(M, N_vel, N_s_w, N_csr) # the default branches, see the paper 

dict_vel_layers = {'conv':[[5,5],[5,5]], 'dense':[100, 70] }
dict_width_layers = {'conv':[[5,5],[5,5]], 'dense':[100, 50] }
dict_csr_layers = {'conv':[[5,5],[5,5]], 'dense':[100, 60] }
model = RadarNet.create_convolutional_network(M, N_vel, N_s_w, N_csr, dict_vel_layers, dict_width_layers, dict_csr_layers)
############################
model.summary()


EPOCHS = 100
BS = 512
lr = 1e-4
H = RadarNet.model_compile_and_train(device,model, X_train, y_train_cat_vel, y_train_cat_s_w, y_train_cat_csr, X_test, y_test_cat_vel, y_test_cat_s_w, y_test_cat_csr, '', EPOCHS, BS, lr)

#Ploting the training and validation metrics
RadarNet.plot_training(H, 'plot_training/')