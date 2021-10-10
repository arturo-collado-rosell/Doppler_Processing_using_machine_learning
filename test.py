#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 00:13:12 2021

@author: acr
"""

import synthetic_weather_data_IQ
import numpy as np
import matplotlib.pyplot as plt

parameters = {'phenom_power':1, 'phenom_vm':5, 'phenom_s_w':2,
              'clutter_power':1000, 'clutter_s_w': 0.25,
              'PRF':1000,
              'noise_power': 1.0e-5}    
data, time = synthetic_weather_data_IQ.synthetic_IQ_data(**parameters)

    
w = np.kaiser(64, 8)
data_w = data * w    

dep = synthetic_weather_data_IQ.DEP_estimation(data_w, w = w, PRF = 1000)
    
    


 
plt.plot(10*np.log10(np.fft.fftshift(dep[0,:])))    
plt.show()