#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 23:07:33 2022

@author: acr
"""

import numpy as np
import tensorflow.keras as keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=512, dim=(158), n_channels=1,
                 n_classes=[26,50,15], shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs) 

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        #list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(index)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim))
        y = np.empty((self.batch_size,3), dtype=int)
        #print(index)
        # Generate data
        aux = np.load('training_data/' + f"{index}_batch" + '.npy')
        X = aux[:, 0:-3]

        # Store class
        y = aux[:, -3: ]
        # for i, ID in enumerate(list_IDs_temp):
        #     # Store sample
        #     aux = np.load('training_data/' + f"{ID}_batch" + '.npy')
        #     X[i,:] = aux[i, 0:-3]

        #     # Store class
        #     y[i,:] = aux[i, -3: ]
        y_csr = keras.utils.to_categorical(y[:,0], num_classes=self.n_classes[0])    
        y_vel = keras.utils.to_categorical(y[:,1], num_classes=self.n_classes[1])
        y_width = keras.utils.to_categorical(y[:,2], num_classes=self.n_classes[2])
        return X, {"velocity_output" :y_vel, "width_output" : y_width, "csr_output" : y_csr}