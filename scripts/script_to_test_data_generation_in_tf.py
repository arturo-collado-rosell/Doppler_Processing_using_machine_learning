#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 02:26:59 2022

@author: acr
"""
from Data_generator_class import DataGenerator
import os
import RadarNet 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

import time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession 
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#number of categories, from the some_params_to_train.npy is extracted this information
# dirName = 'training_data/'
# try:
#     meta_params = np.load(dirName + 'some_params_to_train.npy')
#     N_vel = int(meta_params[0])
#     N_s_w = int(meta_params[1])
#     N_csr = int(meta_params[2])
#     operation_mode = str(meta_params[3])
#     training_data = np.load(dirName + 'training_data.npy')
#     M = training_data.shape[1]-3
#     X = training_data[:,:M].copy()
#     y = training_data[:,M:].copy()
#     del training_data
# except Exception as e:
#     print(e)     
    

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 29)
# #converting the labels to categorical 
# y_train_cat_csr = tf.keras.utils.to_categorical(y_train[:,0], N_csr)
# y_train_cat_vel = tf.keras.utils.to_categorical(y_train[:,1], N_vel)
# y_train_cat_s_w = tf.keras.utils.to_categorical(y_train[:,2], N_s_w)

# y_test_cat_csr = tf.keras.utils.to_categorical(y_test[:,0], N_csr)
# y_test_cat_vel = tf.keras.utils.to_categorical(y_test[:,1], N_vel)
# y_test_cat_s_w = tf.keras.utils.to_categorical(y_test[:,2], N_s_w)
M = 158
N_vel = 50
N_s_w = 15
N_csr = 26
# device = '/CPU:0'
device = '/GPU:0'
# Here you can build your model, every brach separated or the deafult branch networks 

# model = RadarNet.build_all_conv1D(M, N_vel, N_s_w, N_csr) # the default branches, see the paper 

# dict_vel_layers = {'conv':[[5,5],[5,5]], 'dense':[100, 70] }
# dict_width_layers = {'conv':[[5,5],[5,5]], 'dense':[100, 50] }
# dict_csr_layers = {'conv':[[5,5],[5,5]], 'dense':[100, 60] }

dict_vel_layers = {'conv':[[6,20]], 'dense':[150, 100] }
dict_width_layers = {'conv':[[6,20]], 'dense':[150, 80] }
dict_csr_layers = {'conv':[[6,20]], 'dense':[150, 90] }

model = RadarNet.create_convolutional_network(M, N_vel, N_s_w, N_csr, dict_vel_layers, dict_width_layers, dict_csr_layers)
############################
model.summary()

EPOCHS = 2
BS = 512
lr = 1e-4

plot_dir = 'plot_training/'
if not os.path.exists(plot_dir):
     os.mkdir(plot_dir)
     print("Directory " , plot_dir ,  " Created ")
else:    
     print("Directory " , plot_dir ,  " already exists")

directory_to_save_model = plot_dir  +device[1:4]  + '_' + str(EPOCHS) + '_' + str(BS)  
opt = Adam(lr=1e-4) 
EPOCHS = 100
BS = 512

losses = {"velocity_output":"categorical_crossentropy","width_output":"categorical_crossentropy", "csr_output":"categorical_crossentropy"}
lossWeights = {"velocity_output": 1.0, "width_output": 1.0 , "csr_output": 1.0}    


model.compile(optimizer = opt, loss = losses, loss_weights = lossWeights, metrics=["accuracy"])#, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()
custom_early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        min_delta=0.005, 
        mode='auto'
        )

params = {"dim": 158,
          "batch_size": BS,
          "n_classes": (N_csr, N_vel, N_s_w)
          }

training_IDs = [i for i in range(1200)]
validation_IDs = [j for j in range(1200,1451)]

#Generators
training_generator = DataGenerator(training_IDs, **params)
validation_generator = DataGenerator(validation_IDs, **params)

start = time.time()
H = model.fit_generator(training_generator,
              validation_data = validation_generator,
              epochs = EPOCHS,
              use_multiprocessing=True,
              workers = -1,
              verbose = 1,
              callbacks=[custom_early_stopping])
# H = model.fit(X_train,
#               {"velocity_output" : y_train_cat_vel, "width_output" : y_train_cat_s_w, "csr_output" : y_train_cat_csr },
#               validation_data = (X_test, {"velocity_output" :y_test_cat_vel, "width_output" : y_test_cat_s_w, "csr_output" : y_test_cat_csr}),
#               epochs = EPOCHS,
#               batch_size = BS,
#               verbose = 1,
#               callbacks=[custom_early_stopping])
          
end = time.time()
# score = model.evaluate(X_test, {"velocity_output" :y_test_cat_vel, "width_output" : y_test_cat_s_w, "csr_output" : y_test_cat_csr}, verbose=0)
# #Ploting the training and validation metrics
# RadarNet.plot_training(H, plot_dir)

# ########################Predictions to build the class diference histograms #######

# model = tf.keras.models.load_model(directory_to_save_model)

# with tf.device(device):
#     (ypred_vel, ypred_width, ypred_csr) = model.predict(X_test)
#     ypred_vel = np.argmax(ypred_vel, axis = 1)   # select the velocity class
#     ypred_width = np.argmax(ypred_width, axis = 1) # select the width class 
#     ypred_csr = np.argmax(ypred_csr, axis = 1) # select the width class 

# diff_vel = ypred_vel - y_test[:,1]
# diff_width = ypred_width - y_test[:,2]
# diff_csr = ypred_csr - y_test[:,0]

# pd.DataFrame(diff_vel).to_csv(plot_dir + "velocity_diff.csv")
# pd.DataFrame(diff_width).to_csv(plot_dir + "sw_diff.csv")
# pd.DataFrame(diff_csr).to_csv(plot_dir + "CSR_diff.csv")

# import matplotlib.pyplot as plt
# fig = plt.figure() 
# plt.hist(diff_vel)
# plt.xlabel('Velocity error classes ')
# plt.ylabel('Normalized frequency')
# fig.show()

# fig = plt.figure() 
# plt.hist(diff_width)
# plt.xlabel('Spectrum width  error classes ')
# plt.ylabel('Normalized frequency')
# fig.show()

# fig = plt.figure() 
# plt.hist(diff_csr)
# plt.xlabel('CSR error classes ')
# plt.ylabel('Normalized frequency')
# fig.show()