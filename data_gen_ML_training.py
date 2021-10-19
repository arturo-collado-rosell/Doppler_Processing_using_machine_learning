#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 19:35:51 2021

@author: Arturo Collado Rosell
email: arturo.collado.rosell@gmail.com
"""

import numpy as np
import synthetic_weather_data_IQ

# M: data length
#     Fc: radar carrier frecuecy [Hz]
#     Tu: pulse repetition time [s]. In the case of staggered [n1, n2], Tu = T2 - T1
#     theta_3dB_acimut: acimut one-way half power width [degree]
#     radar_mode: radar operation mode, it could be 'uniform' or 'staggered'
#     int_stagg: staggered integers e.g [2, 3]
#     N_vel: velocity classes
#     N_spectral_w: spectral with classes
#     N_csr: clutter to noise ratio classes. It account for the non-clutter class
#     N_snr: signal to noise ratio classes.
#     csr_interval: CSR interval for grid construcction [csr_min, csr_max] e.g [0, 50] [dB]
#     s_w_interval: spectral width interval for grid construcction [s_w_min, s_w_max] * v_a, these two numbers must be fractionals one e.g [0.04, 0.4]*v_a, [m/s] 
#     snr_interval: signal to noise ratio interval [snr_min, snr_max], e.g [0, 30] [dB]
    


Input_params = {'M':64,
                'Fc': 5.6e9,
                'Tu':0.25e-3,
                'theta_3dB_acimut':1,
                'radar_mode':'uniform',
                'L': 2
                }    
    
synthetic_weather_data_IQ.synthetic_data_train(**Input_params) 

