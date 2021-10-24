#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12/04/2021
@author: Arturo Collado Rosell
"""
import time
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

    #################Conv1D architecture##################################
    
def build_conv1D_velocity_branch(inputs, numCategories, finalAct = 'softmax'):
    x = Conv1D(5, 5, activation="relu")(inputs)
    x = Conv1D(5, 5, activation="relu")(x)
    # x = Conv1D(5, 5, activation="relu")(x)
    #x = Conv1D(5, 5, activation="relu")(x)
    # x = Conv1D(5, 5, activation="relu")(x)
    x = Flatten()(x)
    # x = Dense(200, activation="relu")(x)
    # x = Dense(100, activation="relu")(x)
    x = Dense(80, activation="relu")(x)
    x = Dense(60, activation="relu")(x)
    x = Dense(numCategories, activation= finalAct, name = "velocity_output")(x)
    return x
    
      
def build_conv1D_width_branch(inputs, numCategories, finalAct = 'softmax'):
    x = Conv1D(5, 5, activation="relu")(inputs)
    x = Conv1D(5, 5, activation="relu")(x)
    # x = Conv1D(5, 5, activation="relu")(x)
    # x = Conv1D(5, 5, activation="relu")(x)
    # x = Conv1D(5, 5, activation="relu")(x)
    x = Flatten()(x)
    # x = Dense(200, activation="relu")(x)
    # x = Dense(100, activation="relu")(x)
    x = Dense(80, activation="relu")(x)
    x = Dense(40, activation="relu")(x)
    x = Dense(numCategories, activation= finalAct, name = "width_output")(x)
    return x    
    
      
def build_conv1D_csr_branch(inputs, numCategories, finalAct = 'softmax'):
    x = Conv1D(5, 5, activation="relu")(inputs)
    x = Conv1D(5, 5, activation="relu")(x)
    # x = Conv1D(5, 5, activation="relu")(x)
    # x = Conv1D(5, 5, activation="relu")(x)
    # x = Conv1D(5, 5, activation="relu")(x)
    x = Flatten()(x)
    # x = Dense(200, activation="relu")(x)
    # x = Dense(100, activation="relu")(x)
    x = Dense(80, activation="relu")(x)
    x = Dense(50, activation="relu")(x)
    x = Dense(numCategories, activation= finalAct, name = "csr_output")(x)
    return x    
    
        
    
def build_all_conv1D(input_shape, numCategories_velocity, numCategories_width, numCategories_cnr):
    inputs = Input(shape = (input_shape,1))
       
        
    velocity_branch = build_conv1D_velocity_branch(inputs,numCategories_velocity, finalAct = 'softmax')
    width_branch = build_conv1D_width_branch(inputs, numCategories_width, finalAct = 'softmax')
    csr_branch = build_conv1D_csr_branch(inputs, numCategories_cnr, finalAct = 'softmax')
        
    model = Model(
                inputs = inputs,
                outputs = [velocity_branch, width_branch, csr_branch],
                name = "radarnet_conv1d")
    return model
    
def model_compile_and_train(model, X_train, y_train_cat_vel, y_train_cat_s_w, y_train_cat_csr, X_test, y_test_cat_vel, y_test_cat_s_w, y_test_cat_csr):
    #compiling the model
    opt = Adam(lr=1e-4) 
    EPOCHS = 10
    BS = 512
    
    losses = {"velocity_output":"categorical_crossentropy","width_output":"categorical_crossentropy", "csr_output":"categorical_crossentropy"}
    lossWeights = {"velocity_output": 1.0, "width_output": 1.0 , "csr_output": 1.0}    

    
    model.compile(optimizer = opt, loss = losses, loss_weights = lossWeights, metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    custom_early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            min_delta=0.005, 
            mode='auto'
            )

    start = time.time()
    H = model.fit(X_train,
             {"velocity_output" : y_train_cat_vel, "width_output" : y_train_cat_s_w, "csr_output" : y_train_cat_csr },
                  validation_data = (X_test, {"velocity_output" :y_test_cat_vel, "width_output" : y_test_cat_s_w, "csr_output" : y_test_cat_csr}),
                  epochs = EPOCHS,
                  batch_size = BS,
                  verbose = 1,
                  callbacks=[custom_early_stopping])
                  
    end = time.time()
    elapsed_time = end - start
    print('The trainig time was {} seconds'.format(elapsed_time))    
    return H