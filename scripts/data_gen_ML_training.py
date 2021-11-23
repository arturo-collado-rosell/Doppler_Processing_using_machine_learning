#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 19:35:51 2021

With this script you will generate the data to train the NN. Two file are saved, the first one with the PSDs and the other with some parameters
used for data generation


@author: Arturo Collado Rosell
email: arturo.collado.rosell@gmail.com
"""
import os
import numpy as np
import synthetic_weather_data_IQ


Input_params = {'M':64,
                'Fc': 5.6e9,
                'Tu':0.25e-3,
                'theta_3dB_acimut':1,
                'radar_mode':'staggered',
                'L': 10
                }    
    
data_PSD, N_vel, N_s_w, N_csr, radar_mode = synthetic_weather_data_IQ.synthetic_data_train(**Input_params) 

dirName = 'training_data/'
if not os.path.exists(dirName):
     os.mkdir(dirName)
     print("Directory " , dirName ,  " Created ")
else:    
     print("Directory " , dirName ,  " already exists")
np.save(dirName + 'training_data', data_PSD) 
np.save(dirName + 'some_params_to_train',(N_vel, N_s_w, N_csr, radar_mode)) 

