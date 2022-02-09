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
    

    
def model_compile_and_train(device, model, X_train, y_train_cat_vel, y_train_cat_s_w, y_train_cat_csr, X_test, y_test_cat_vel, y_test_cat_s_w, y_test_cat_csr, directory_to_save, EPOCHS = 100, BS = 512, lr = 1e-4):
    with tf.device(device):
        #compiling the model
        opt = Adam(lr=lr) 
        EPOCHS = EPOCHS
        BS = BS
    
        losses = {"velocity_output":"categorical_crossentropy","width_output":"categorical_crossentropy", "csr_output":"categorical_crossentropy"}
        lossWeights = {"velocity_output": 1.0, "width_output": 1.0 , "csr_output": 1.0}    

    
        model.compile(optimizer = opt, loss = losses, loss_weights = lossWeights, metrics=["accuracy"])#, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()
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
        # Saving the model
        model.save(directory_to_save + 'model.h5')
        elapsed_time = end - start
        print('The trainig time was {} seconds'.format(elapsed_time))    
        return H

def plot_training( H, directory_to_save, plot = True):
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
    plt.savefig(directory_to_save + "{}_loss.png".format('train_and_validate'))
    if plot == True:
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
    plt.savefig(directory_to_save  + "{}_accs.png".format('train_and_validate'))
    if plot == True:
        plt.show() 

# #create a new figure for the Precision
#     precisionNames = ["velocity_output_precision_2", "width_output_precision_2", "csr_output_precision_2"]
#     plt.style.use("ggplot")
#     (fig, ax) = plt.subplots(3, 1, figsize=(8, 8))
     
#     Precision_t_vector = []
#     Precision_v_vector = []
#     # loop over the accuracy names
#     for (i, l) in enumerate(precisionNames):
        
#      	# plot the loss for both the training and validation data
#         ax[i].set_title("Precision for {}".format(l))
#         ax[i].set_xlabel("Epoch #")
#         ax[i].set_ylabel("Precision")
#         ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
#         ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l], label="val_" + l)
#         ax[i].legend()
#         Precision_t_vector.append(H.history[l])
#         Precision_v_vector.append(H.history["val_" + l])
     
#     # save the accuracies figure
#     plt.tight_layout()
#     plt.savefig(directory + "{}_precision.png".format('train_and_validate'))
#     plt.show()
# # create a new figure for the Recall
#     recallNames = ["velocity_output_recall_2", "width_output_recall_2", "csr_output_recall_2"]
#     plt.style.use("ggplot")
#     (fig, ax) = plt.subplots(3, 1, figsize=(8, 8))
     
#     Recall_t_vector = []
#     Recall_v_vector = []
#     # loop over the accuracy names
#     for (i, l) in enumerate(recallNames):
        
#      	# plot the loss for both the training and validation data
#         ax[i].set_title("Recall for {}".format(l))
#         ax[i].set_xlabel("Epoch #")
#         ax[i].set_ylabel("Recall")
#         ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
#         ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l], label="val_" + l)
#         ax[i].legend()
#         Recall_t_vector.append(H.history[l])
#         Recall_v_vector.append(H.history["val_" + l])
     
#     # save the accuracies figure
#     plt.tight_layout()
#     plt.savefig(directory + "{}_recall.png".format('train_and_validate'))    
#     plt.show()
    
    #save training and validation loss and accuracy
    pd.DataFrame( Loss_t_vector).to_csv(directory_to_save +'Losstraining' + '.csv')
    pd.DataFrame( Loss_v_vector).to_csv(directory_to_save+'Lossvalidation' + '.csv')
    pd.DataFrame( Accuracy_t_vector).to_csv(directory_to_save+'Accuracytraining' + '.csv')
    pd.DataFrame( Accuracy_v_vector).to_csv(directory_to_save+'Accuracyvalidation' + '.csv')
    
def prediction(model, data_PSD, device = 'CPU:/0'):
    """

    
    """
    with tf.device(device):
        
        start = time.time()  
        (y_pred_vel, y_pred_sw, y_pred_csr) = model.predict(data_PSD)
        y_pred_vel = np.argmax(y_pred_vel, axis = 1)
        y_pred_sw = np.argmax(y_pred_sw, axis = 1)
        y_pred_csr = np.argmax(y_pred_csr, axis = 1)
        end = time.time()
        elapsed_time = end - start
        
        return y_pred_vel, y_pred_sw, y_pred_csr, elapsed_time
    

def build_conv1D_vel_branch(inputs, numCategories, dict_vel_layers, finalAct = 'softmax'):   
    conv_layers = dict_vel_layers['conv']
    dense_layers = dict_vel_layers['dense']    
    x = Conv1D(conv_layers[0][0],conv_layers[0][1], activation="relu")(inputs)
    for l in conv_layers[1:]:
        x = Conv1D(l[0], l[1], activation="relu")(x)       
    x = Flatten()(x)    
    for l in dense_layers:
        x = Dense(l, activation="relu")(x)   
    x = Dense(numCategories, activation= finalAct, name = "velocity_output")(x)
    return x

    
def build_conv1D_width1_branch(inputs, numCategories, dict_width_layers, finalAct = 'softmax'):   
    conv_layers = dict_width_layers['conv']
    dense_layers = dict_width_layers['dense']    
    x = Conv1D(conv_layers[0][0],conv_layers[0][1], activation="relu")(inputs)
    for l in conv_layers[1:]:
        x = Conv1D(l[0], l[1], activation="relu")(x)       
    x = Flatten()(x)    
    for l in dense_layers:
        x = Dense(l, activation="relu")(x)   
    x = Dense(numCategories, activation= finalAct, name = "width_output")(x)
    return x
    
    
def build_conv1D_csr1_branch(inputs, numCategories, dict_csr_layers, finalAct = 'softmax'):   
    conv_layers = dict_csr_layers['conv']
    dense_layers = dict_csr_layers['dense']    
    x = Conv1D(conv_layers[0][0],conv_layers[0][1], activation="relu")(inputs)
    for l in conv_layers[1:]:
        x = Conv1D(l[0], l[1], activation="relu")(x)       
    x = Flatten()(x)    
    for l in dense_layers:
        x = Dense(l, activation="relu")(x)   
    x = Dense(numCategories, activation= finalAct, name = "csr_output")(x)
    return x
    
    

def create_convolutional_network(input_shape, numCategories_velocity, numCategories_width, numCategories_cnr, dict_vel_layers, dict_sw_layers, dict_csr_layers):
    inputs = Input(shape = (input_shape,1))
       
        
    velocity_branch = build_conv1D_vel_branch(inputs,numCategories_velocity, dict_vel_layers, finalAct = 'softmax')
    width_branch = build_conv1D_width1_branch(inputs, numCategories_width, dict_sw_layers , finalAct = 'softmax')
    csr_branch = build_conv1D_csr1_branch(inputs, numCategories_cnr, dict_csr_layers, finalAct = 'softmax')
        
    model = Model(
                inputs = inputs,
                outputs = [velocity_branch, width_branch, csr_branch],
                name = "radarnet_conv1d")
    return model








    