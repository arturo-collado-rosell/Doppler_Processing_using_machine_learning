#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 19:35:51 2021

@author: Arturo Collado Rosell
email: arturo.collado.rosell@gmail.com
"""

import numpy as np
import synthetic_weather_data_IQ



np.random.seed(2021) # seed for reproducibility 
M = 64
Fc = 5.6e9
wavelenght = 3e8/Fc
Tu = 0.25e-3
v_a = wavelenght/(4.0*Tu)
theta_3dB_acimut = 1 # [degree]

radar_mode = 'uniform' 

# specific parameters dependent of the operational mode
if radar_mode == 'uniform':    
    # radar rotation speed    
    w =  theta_3dB_acimut * np.pi/180.0 / (M * Tu)
    N_vel = 40
    N_spectral_w = 12
    N_CSR = 25
    
 
elif radar_mode == 'staggered':
    int_stagg = [2, 3]
    # radar rotation speed
    w =  theta_3dB_acimut * np.pi/180.0 / (M * sum(int_stagg)/len(int_stagg)* Tu)
    N_vel = 50
    N_spectral_w = 10
    N_CSR = 25
else:
    raise Exception('The radar operation mode is not valid')

    
clutter_spectral_width = w*wavelenght * np.sqrt(np.log(2.0)) / (2.0 * np.pi * theta_3dB_acimut* np.pi/180.0) # [m/s]
Sp = 1 # precipitation power


######### signal parameters intervals ################################## 
csr = np.linspace(0,50,25)
csr = np.concatenate((np.zeros(8,), csr))

