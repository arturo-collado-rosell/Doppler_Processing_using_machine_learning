#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 17:24:33 2021

@author: Arturo Collado Rosell

This script is to reproduce one of the experiment of the original paper, the one varying 
the CSR

"""
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'machine_learning_scripts/')
import matplotlib.pyplot as plt
import RadarNet 

import numpy as np
import tensorflow as tf
import synthetic_weather_data_IQ

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession 
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

Tu = 0.25e-3
PRF = 1/Tu
M = 64
Fc = 5.6e9
c = 3.0e8
theta_3dB_acimut = 1
wavelenght = c/Fc
va = wavelenght*PRF/4
Sp = 1
vm = [0.2*va, 0.4*va]
spectral_w = [3, 5]
csr = np.linspace(0, 50, 25)
Sc = Sp * 10**(csr/10)

radar_mode = 'staggered'
int_stagg = [2, 3]

snr = 20
power_noise = Sp / (10**(snr/10))

I = 1

w =  theta_3dB_acimut * np.pi/180.0 / (M * sum(int_stagg)/len(int_stagg)* Tu)
clutter_spectral_width = w*wavelenght * np.sqrt(np.log(2.0)) / (2.0 * np.pi * theta_3dB_acimut* np.pi/180.0) # [m/s]

#         clutter_power: power of clutter [times]
#         clutter_s_w : clutter spectrum width [m/s] 
#         phenom_power: power of phenom [times] 
#         phenom_vm: phenom mean Doppler velocity [m/s]
#         phenom_s_w: phenom spectrum width [m/s]
#         noise_power: noise power [times] 
#         M: number of samples in a CPI, the sequence length  
#         wavelength: radar wavelength [m] 
#         PRF: Pulse Repetition Frequecy [Hz]
#         radar_mode: Can be "uniform" or "staggered" 
#         int_stagg: list with staggered sequence, by default it is [2,3] when radar_mode == "staggered" 
#         samples_factor: it needed for windows effects, by default it is set to 10
#         num_realizations: number of realization 


parameters_with_clutter = {    'clutter_s_w': clutter_spectral_width,
                               'phenom_power':1,        
                               'noise_power': power_noise,   
                               'PRF':PRF,
                               'radar_mode':radar_mode,
                               'M':M,
                               'wavelenght':wavelenght,
                               'int_stagg':int_stagg,
                               'num_realizations':I
                               }   
#windows
num_samples_uniform = round((M - 1)* sum(int_stagg)/len(int_stagg)) + 1
window = np.kaiser(num_samples_uniform, 8)


###########Data generation#####################################
L = 1000 # Monte Carlo realization
N_vel = len(vm)
N_Sc = len(Sc)
N_sw = len(spectral_w) 
data_PSD = np.zeros((N_vel*N_Sc*N_sw*L, num_samples_uniform))
for i in range(N_vel):
    for q in range(N_Sc):
        for j in range(N_sw):
            parameters_with_clutter['phenom_vm'] = vm[i]
            parameters_with_clutter['clutter_power'] = Sc[q]
            parameters_with_clutter['phenom_s_w'] = spectral_w[j]
            for ind_l in range(L):
                z_IQ, _ = synthetic_weather_data_IQ.synthetic_IQ_data(**parameters_with_clutter)
                        
                if radar_mode == 'staggered':
                    z_IQ_zero_interp = synthetic_weather_data_IQ.zero_interpolation(z_IQ, int_stagg)
                    data_w = z_IQ_zero_interp * window
                else:
                    data_w = z_IQ * window
                        
                ##PSD estimation
                psd = synthetic_weather_data_IQ.PSD_estimation(data_w, w = window, PRF = 1/Tu)
                psd = psd /np.max(psd)
                data_PSD[i*N_Sc*N_sw*L + q*N_sw*L + j*L + ind_l,:] = 10*np.log10(psd[0,:])
                
################################################################                
                
#predictions using the NN
model = tf.keras.models.load_model('GPU_3_512model.h5')
vel_pred, sw_pred, csr_pred = RadarNet.prediction(model, data_PSD, device = '/GPU:0')                

try:
    meta_params = np.load('some_params_to_train.npy')
    N_vel_grid = int(meta_params[0])
    N_s_w_grid = int(meta_params[1])
    N_csr_grid = int(meta_params[2])
except Exception as e:
    print(e)  
    
# Clutter power grid
csr_g = np.linspace(0, 50, N_csr_grid-1)
Sc_grid =  Sp * 10.0**(csr_g/10)
# spectral width grid
s_width_grid = np.linspace(0.04, 0.2, N_s_w_grid) * va
# velocity grid
vel_step = 2.0/N_vel_grid * va
vel_grid = -np.arange(-va + vel_step/2.0, va , vel_step)    


vel_pred = vel_grid[vel_pred]
sw_pred = s_width_grid[sw_pred] 

vel_NN = np.zeros((N_sw, N_Sc, N_vel))
std_vel_NN = np.zeros((N_sw, N_Sc, N_vel))
sw_NN = np.zeros((N_sw, N_Sc, N_vel))
std_sw_NN = np.zeros((N_sw, N_Sc, N_vel))
   
for i in range(N_vel):
    for q in range(N_Sc):
        for j in range(N_sw):
            batch_vel = vel_pred[i*N_Sc*N_sw*L + q*N_sw*L + j*L: i*N_Sc*N_sw*L + q*N_sw*L + (j+1)*L]
            batch_sw = sw_pred[i*N_Sc*N_sw*L + q*N_sw*L + j*L: i*N_Sc*N_sw*L + q*N_sw*L + (j+1)*L]
            
            vel_NN[j,q,i] = np.mean(batch_vel)
            std_vel_NN[j,q,i] = np.std(batch_vel)
            sw_NN[j,q,i] = np.mean(batch_sw)
            std_sw_NN[j,q,i] = np.std(batch_sw)
            
###Ploting the results
fig, ax = plt.subplots()
ax.plot(csr, vel_NN[0,:,0] - vm[0]*np.ones((N_Sc,)) )
ax.grid()
plt.show()
            
