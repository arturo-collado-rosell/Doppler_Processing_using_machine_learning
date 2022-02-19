#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 22:45:53 2022
script to make predictions and to plot and save NN training after Optuna study

@author: Arturo Collado Rosell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

optuna_dir = 'models/'

dirName = 'training_data/'

try:
    meta_params = np.load(dirName + 'some_params_to_train.npy')
    N_vel = int(meta_params[0])
    N_s_w = int(meta_params[1])
    N_csr = int(meta_params[2])
    training_data = np.load(dirName + 'training_data.npy')
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

trial_number = 14
model = tf.keras.models.load_model(optuna_dir + f'_{trial_number}.h5')
device = '/GPU:0'
with tf.device(device):
    
    (ypred_vel, ypred_width, ypred_csr) = model.predict(X_test)
    ypred_vel = np.argmax(ypred_vel, axis = 1)   # select the velocity class
    ypred_width = np.argmax(ypred_width, axis = 1) # select the width class 
    ypred_csr = np.argmax(ypred_csr, axis = 1) # select the width class
    
diff_vel = ypred_vel - y_test[:,1]
diff_width = ypred_width - y_test[:,2]
diff_csr = ypred_csr - y_test[:,0]

pd.DataFrame(diff_vel).to_csv(optuna_dir + f"{trial_number}_velocity_diff.csv")
pd.DataFrame(diff_width).to_csv(optuna_dir + f"{trial_number}_sw_diff.csv")
pd.DataFrame(diff_csr).to_csv(optuna_dir + f"{trial_number}_CSR_diff.csv")


fig = plt.figure() 
plt.hist(diff_vel)
plt.xlabel('Velocity error classes ')
plt.ylabel('Normalized frequency')
fig.show()

fig = plt.figure() 
plt.hist(diff_width)
plt.xlabel('Spectrum width  error classes ')
plt.ylabel('Normalized frequency')
fig.show()

fig = plt.figure() 
plt.hist(diff_csr)
plt.xlabel('CSR error classes ')
plt.ylabel('Normalized frequency')
fig.show()    
                 
H = pd.read_csv(optuna_dir + f'_{trial_number}.csv')

######
lossNames = ["loss", "velocity_output_loss", "width_output_loss", "csr_output_loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(4, 1, figsize=(8, 8))
EPOCHS = len(H["loss"]) 
    
Loss_t_vector = []
Loss_v_vector = []
# loop over the loss names
for (i, l) in enumerate(lossNames):
    Loss_t_vector.append(H[l])
    Loss_v_vector.append(H["val_" + l])
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(np.arange(0, EPOCHS), H[l], label=l)
    ax[i].plot(np.arange(0, EPOCHS), H["val_" + l], label="val_" + l)
    ax[i].legend()
    # plot the loss for both the training and validation data
plt.tight_layout()
plt.show()


accuracyNames = ["velocity_output_accuracy", "width_output_accuracy", "csr_output_accuracy"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(8, 8))
     
Accuracy_t_vector = []
Accuracy_v_vector = []
# loop over the accuracy names
for (i, l) in enumerate(accuracyNames):
    #plot the loss for both the training and validation data
    ax[i].set_title("Accuracy for {}".format(l))
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Accuracy")
    ax[i].plot(np.arange(0, EPOCHS), H[l], label=l)
    ax[i].plot(np.arange(0, EPOCHS), H["val_" + l], label="val_" + l)
    ax[i].legend()
    Accuracy_t_vector.append(H[l])
    Accuracy_v_vector.append(H["val_" + l])
     
    # save the accuracies figure
plt.tight_layout()
plt.show()

pd.DataFrame( Loss_t_vector).to_csv(optuna_dir + f'{trial_number}_Losstraining' + '.csv')
pd.DataFrame( Loss_v_vector).to_csv(optuna_dir + f'{trial_number}_Lossvalidation' + '.csv')
pd.DataFrame( Accuracy_t_vector).to_csv(optuna_dir +f'{trial_number}_Accuracytraining' + '.csv')
pd.DataFrame( Accuracy_v_vector).to_csv(optuna_dir + f'{trial_number}_Accuracyvalidation' + '.csv')