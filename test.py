#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 00:13:12 2021

@author: Arturo Collado Rosell
email: arturo.collado.rosell@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
import synthetic_weather_data_IQ


parameters_with_clutter = {'phenom_power':1, 'phenom_vm':10, 'phenom_s_w':2,
              'clutter_power':1000, 'clutter_s_w': 0.25,
              'PRF':4000,
              'noise_power': 1.0e-5,
              'radar_mode':'staggered'}   

parameters_phenom = {'phenom_power':1, 'phenom_vm':10, 'phenom_s_w':2,
              'PRF':4000,
              'noise_power': 1.0e-5,
              'radar_mode':'staggered'}   

parameters = parameters_phenom
#data generation
data, time = synthetic_weather_data_IQ.synthetic_IQ_data(**parameters)

    

if parameters['radar_mode'] == 'staggered':
    data_zero_interp = synthetic_weather_data_IQ.zero_interpolation(data, [2,3])
    w = np.kaiser(data_zero_interp.shape[1], 8)
    data_w = data_zero_interp * w
    
else:
    w = np.kaiser(64, 8)
    data_w = data * w 

# DEP estimation using periodogram
dep = synthetic_weather_data_IQ.DEP_estimation(data_w, w = w, PRF = parameters['PRF'])
    
    

# Plotting the DEP 
plt.plot(10*np.log10(np.fft.fftshift(dep[0,:])))    
plt.show()