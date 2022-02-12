#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:55:21 2022

@author: acr
"""

#######Optuna study#############
from Data_generator_class import DataGenerator
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import optuna 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.backend import clear_session


dirName = 'training_data/'
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


device = '/GPU:0'

EPOCHS = 120

dirName_models = 'models/'
if not os.path.exists(dirName_models):
     os.mkdir(dirName_models)
     print("Directory " , dirName_models ,  " Created ")
else:    
     print("Directory " , dirName_models ,  " already exists")

def create_model(trial):
  n_conv_layers = trial.suggest_int("n_layers_conv",1,4)
  n_dens_layers = trial.suggest_int("n_layers_dens",1,4)

  input_layer = Input(shape = (M,1)) #input layer
  x_vel = input_layer
  x_width = input_layer
  x_csr = input_layer
  for i in range(n_conv_layers):
    x_vel = Conv1D(filters=trial.suggest_categorical(f'filters_vel_{i}', [2, 5, 10, 15]), kernel_size = trial.suggest_categorical(f"kernel_size_vel_{i}", [5, 10, 15]), activation = 'relu')(x_vel)
    x_width = Conv1D(filters=trial.suggest_categorical(f'filters_width_{i}', [2, 5, 10, 15]), kernel_size = trial.suggest_categorical(f"kernel_size_width_{i}", [5, 10, 15]), activation = 'relu')(x_width)
    x_csr = Conv1D(filters=trial.suggest_categorical(f'filters_csr_{i}', [2, 5, 10, 15]), kernel_size = trial.suggest_categorical(f"kernel_size_csr_{i}", [5, 10, 15]), activation = 'relu')(x_csr)

  x_vel = Flatten()(x_vel)
  x_width = Flatten()(x_width)
  x_csr = Flatten()(x_csr)
  for i in range(n_dens_layers):
    x_vel = Dense(units = trial.suggest_categorical(f"units_vel_{i}", [30,40, 50, 60]), activation='relu')(x_vel)
    x_width = Dense(units = trial.suggest_categorical(f"units_width_{i}", [30,40, 50, 60]), activation='relu')(x_width)
    x_csr = Dense(units = trial.suggest_categorical(f"units_csr_{i}", [30,40, 50, 60]), activation='relu')(x_csr)
  
  output_layer_vel = Dense(N_vel, activation='softmax', name = "velocity_output")(x_vel)
  output_layer_width = Dense(N_s_w, activation='softmax', name = "width_output")(x_width)
  output_layer_csr = Dense(N_csr, activation='softmax', name = "csr_output")(x_csr)
  model = Model(inputs = input_layer  , outputs = [output_layer_vel, output_layer_width, output_layer_csr]  , name = 'Doppler_processing_staggered')
  return model

training_IDs = [i for i in range(int(number_of_batch*0.8))]
validation_IDs = [j for j in range(int(number_of_batch*0.8),number_of_batch)]


#Generators
params = {"dim": M,
          "batch_size": batch_size,
          "n_classes": (N_csr, N_vel, N_s_w)
          }
training_generator = DataGenerator(training_IDs, **params)
validation_generator = DataGenerator(validation_IDs, **params)

def objective(trial):
    clear_session()
    
    model = create_model(trial)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log = True)
    with tf.device(device):
        
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        custom_early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            min_delta=0.005, 
            mode='auto',
            restore_best_weights = True
            )

        H = model.fit(training_generator,
                      validation_data = validation_generator,
                      epochs = EPOCHS,
                      use_multiprocessing=True,
                      workers = -1,
                      verbose = 1,
                      callbacks=[custom_early_stopping])
        score = model.evaluate(validation_generator)
        model.save(dirName_models + f'_{trial.number}.h5' )
        
        H_df = pd.DataFrame(H.history)
        H_df.to_csv(dirName_models + f'_{trial.number}' + '.csv')
    
    return score[4], score[5], score[6]

if __name__ == "__main__":
    study = optuna.create_study(directions=["maximize", "maximize", "maximize"])
    study.optimize(objective, n_trials= 50 )
    best_params = study.best_params
    df = study.trials_dataframe()
    df.head()
    df.to_csv(dirName_models + "optuna_study.csv")
    print(best_params)    
    