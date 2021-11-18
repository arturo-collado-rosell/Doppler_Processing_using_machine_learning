#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 19:35:51 2021

@author: Arturo Collado Rosell
email: arturo.collado.rosell@gmail.com
"""

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
np.save('training_data', data_PSD) 
np.save('some_params_to_train',(N_vel, N_s_w, N_csr, radar_mode)) 
del data_PSD
