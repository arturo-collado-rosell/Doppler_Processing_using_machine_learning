#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12/04/2021
@author: Arturo Collado Rosell
"""
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
    
def model_compile_and_train(device, model, X_train, y_train_cat_vel, y_train_cat_s_w, y_train_cat_csr, X_test, y_test_cat_vel, y_test_cat_s_w, y_test_cat_csr):
    with tf.device(device):
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

def plot_training( H, directory):
    # plot the total loss, category loss, and color loss
    lossNames = ["loss", "velocity_output_loss", "width_output_loss", "csr_output_loss"]
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(4, 1, figsize=(8, 8))
    EPOCHS = len(H.history["loss"]) 
    
    Loss_t_vector = []
    Loss_v_vector = []
    # loop over the loss names
    for (i, l) in enumerate(lossNames):
        # plot the loss for both the training and validation data
        Loss_t_vector.append(H.history[l])
        Loss_v_vector.append(H.history["val_" + l])
        title = "Loss for {}".format(l) if l != "loss" else "Total loss"
        ax[i].set_title(title)
        ax[i].set_xlabel("Epoch #")
        ax[i].set_ylabel("Loss")
        ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
        ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l], label="val_" + l)
        ax[i].legend()
    	
    	
        
     
    # save the losses figure and create a new figure for the accuracies
    plt.tight_layout()
    plt.savefig(directory + "{}_loss.png".format('train_and_validate'))
    plt.show()
    
    
    # create a new figure for the accuracies
    accuracyNames = ["velocity_output_accuracy", "width_output_accuracy", "csr_output_accuracy"]
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(3, 1, figsize=(8, 8))
     
    Accuracy_t_vector = []
    Accuracy_v_vector = []
    # loop over the accuracy names
    for (i, l) in enumerate(accuracyNames):
        
    	# plot the loss for both the training and validation data
        ax[i].set_title("Accuracy for {}".format(l))
        ax[i].set_xlabel("Epoch #")
        ax[i].set_ylabel("Accuracy")
        ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
        ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l], label="val_" + l)
        ax[i].legend()
        Accuracy_t_vector.append(H.history[l])
        Accuracy_v_vector.append(H.history["val_" + l])
     
    # save the accuracies figure
    plt.tight_layout()
    plt.savefig(directory  + "{}_accs.png".format('train_and_validate'))
    plt.show()

#create a new figure for the Precision
    precisionNames = ["velocity_output_precision", "width_output_precision", "csr_output_precision"]
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(3, 1, figsize=(8, 8))
     
    Precision_t_vector = []
    Precision_v_vector = []
    # loop over the accuracy names
    for (i, l) in enumerate(precisionNames):
        
     	# plot the loss for both the training and validation data
        ax[i].set_title("Precision for {}".format(l))
        ax[i].set_xlabel("Epoch #")
        ax[i].set_ylabel("Precision")
        ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
        ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l], label="val_" + l)
        ax[i].legend()
        Precision_t_vector.append(H.history[l])
        Precision_v_vector.append(H.history["val_" + l])
     
    # save the accuracies figure
    plt.tight_layout()
    plt.savefig(directory + "{}_precision.png".format('train_and_validate'))
    plt.show()
# create a new figure for the Recall
    recallNames = ["velocity_output_recall", "width_output_recall", "csr_output_recall"]
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(3, 1, figsize=(8, 8))
     
    Recall_t_vector = []
    Recall_v_vector = []
    # loop over the accuracy names
    for (i, l) in enumerate(recallNames):
        
     	# plot the loss for both the training and validation data
        ax[i].set_title("Recall for {}".format(l))
        ax[i].set_xlabel("Epoch #")
        ax[i].set_ylabel("Recall")
        ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
        ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l], label="val_" + l)
        ax[i].legend()
        Recall_t_vector.append(H.history[l])
        Recall_v_vector.append(H.history["val_" + l])
     
    # save the accuracies figure
    plt.tight_layout()
    plt.savefig(directory + "{}_recall.png".format('train_and_validate'))    
    plt.show()
    
    #save training and validation loss and accuracy
    # pd.DataFrame( Loss_t_vector).to_csv(directory +'Losstraining' + '.csv')
    # pd.DataFrame( Loss_v_vector).to_csv(directory+'Lossvalidation' + '.csv')
    # pd.DataFrame( Accuracy_t_vector).to_csv(directory+'Accuracytraining' + '.csv')
    # pd.DataFrame( Accuracy_v_vector).to_csv(directory+'Accuracyvalidation' + '.csv')