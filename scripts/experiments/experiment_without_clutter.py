#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:27:51 2021

@author: Arturo Collado Rosell
"""

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')
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

np.random.seed(2021) # seed for reproducibility 

######Simulation Parameers#####################################################
Tu = 0.25e-3
PRF = 1/Tu
M = 64
Fc = 5.6e9
c = 3.0e8
wavelenght = c/Fc
va = wavelenght*PRF/4
Sp = 1
vm = 12
spectral_w = np.linspace(0.04*va, 0.2*va, 20)
csr = np.linspace(0, 50, 25)
Sc = Sp * 10**(csr/10)

radar_mode = 'staggered'
int_stagg = [2, 3]

snr = np.array([10, 15, 20])
power_noise = Sp / (10**(snr/10))

I = 1


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


parameters_no_clutter = {       'phenom_vm':vm,
                               'phenom_power':1,                        
                               'PRF':PRF,
                               'radar_mode':radar_mode,
                               'M':M,
                               'wavelenght':wavelenght,
                               'int_stagg':int_stagg,
                               'num_realizations':I
                               }   
#window
num_samples_uniform = round((M - 1)* sum(int_stagg)/len(int_stagg)) 
window = np.kaiser(num_samples_uniform, 8)

###########Data generation#####################################
L = 1000 # Monte Carlo realization number
N_snr = len(snr)
N_sw = len(spectral_w) 
complex_IQ_data = np.zeros((N_snr*N_sw*L, M), dtype = complex)
data_PSD = np.zeros((N_snr*N_sw*L, num_samples_uniform))
for i in synthetic_weather_data_IQ.progressbar(range(N_snr), 'Computing:') :
        parameters_no_clutter['noise_power'] =  power_noise[i]
        for j in range(N_sw):            
            parameters_no_clutter['phenom_s_w'] = spectral_w[j]
            for ind_l in range(L):
                z_IQ, _ = synthetic_weather_data_IQ.synthetic_IQ_data(**parameters_no_clutter)
                complex_IQ_data[i*N_sw*L + j*L + ind_l,:] = z_IQ        
                if radar_mode == 'staggered':
                    z_IQ_zero_interp = synthetic_weather_data_IQ.zero_interpolation(z_IQ, int_stagg)
                    data_w = z_IQ_zero_interp * window
                else:
                    data_w = z_IQ * window
                        
                ##PSD estimation
                psd = synthetic_weather_data_IQ.PSD_estimation(data_w, w = window, PRF = 1/Tu)
                psd = psd /np.max(psd)
                data_PSD[i*N_sw*L + j*L + ind_l,:] = 10*np.log10(psd[0,:])
                
#predictions using the NN

model = tf.keras.models.load_model('../plot_training/'+'GPU_100_512model.h5')
vel_pred, sw_pred, csr_pred, time = RadarNet.prediction(model, data_PSD, device = '/GPU:0')                

try:
    meta_params = np.load('../training_data/some_params_to_train.npy')
    N_vel_grid = int(meta_params[0])
    N_s_w_grid = int(meta_params[1])
except Exception as e:
    print(e)  
    


# spectral width grid
s_width_grid = np.linspace(0.04, 0.2, N_s_w_grid) * va
# velocity grid
vel_step = 2.0/N_vel_grid * va
vel_grid = -np.arange(-va + vel_step/2.0, va , vel_step)    


vel_pred = vel_grid[vel_pred]
sw_pred = s_width_grid[sw_pred] 


vel_NN = np.zeros((N_sw, N_snr))
std_vel_NN = np.zeros((N_sw, N_snr))
sw_NN = np.zeros((N_sw, N_snr))
std_sw_NN = np.zeros((N_sw, N_snr))
   
for i in range(N_snr):
        for j in range(N_sw):
            batch_vel = vel_pred[i*N_sw*L +  j*L: i*N_sw*L +  (j+1)*L]
            batch_sw = sw_pred[i*N_sw*L +  j*L: i*N_sw*L +  (j+1)*L]
            
            #Bias and standard deviation
            vel_NN[j,i], std_vel_NN[j,i] = np.mean(batch_vel), np.std(batch_vel)             
            sw_NN[j,i], std_sw_NN[j,i] = np.mean(batch_sw), np.std(batch_sw)

###Ploting the results##############################################################

#Velocity Bias
graph = plt.figure(figsize = (7,4))
plt.plot(spectral_w, vel_NN[:,0] - vm*np.ones((N_sw,)),'-.', label = r'SNR= {} dB '.format(snr[0]))
plt.plot(spectral_w, vel_NN[:,1] - vm*np.ones((N_sw,)), '-o' , label = r'SNR= {} dB '.format(snr[1]))
plt.plot(spectral_w, vel_NN[:,2] - vm*np.ones((N_sw,)), '-p', label = r'SNR= {} dB '.format(snr[2]))
plt.grid()
plt.xlabel(r'$\sigma_p$ [m/s]')
plt.ylabel(r'Bias $\hat{v}_p$ [m/s]')
plt.legend()
graph.show()

#Velocity Std
graph = plt.figure(figsize = (7,4))
plt.plot(spectral_w, std_sw_NN[:,0],'-.', label = r'SNR= {} dB '.format(snr[0]))
plt.plot(spectral_w, std_vel_NN[:,1], '-o' , label = r'SNR= {} dB '.format(snr[1]))
plt.plot(spectral_w, std_vel_NN[:,2], '-p', label = r'SNR= {} dB '.format(snr[2]))
plt.grid()
plt.xlabel(r'$\sigma_p$ [m/s]')
plt.ylabel(r'Std $\hat{v}_p$ [m/s]')
plt.legend()
graph.show()



#spectrum width Bias
graph = plt.figure(figsize = (7,4))
plt.plot(spectral_w, sw_NN[:,0] - spectral_w, '-.', label = r'SNR= {} dB '.format(snr[0]) )
plt.plot(spectral_w, sw_NN[:,1] - spectral_w, '-o', label = r'SNR= {} dB '.format(snr[1]) )
plt.plot(spectral_w, sw_NN[:,2] - spectral_w, '-p', label = r'SNR= {} dB '.format(snr[2]) )
plt.grid()
plt.xlabel(r'$\sigma_p$ [m/s]')
plt.ylabel(r'Bias $\hat{\sigma}_p$ [m/s]')
plt.legend()
graph.show()



#spectrum width Std
graph = plt.figure(figsize = (7,4))
plt.plot(spectral_w, std_sw_NN[:,0], '-.', label = r'SNR= {} dB '.format(snr[0]) )
plt.plot(spectral_w, std_sw_NN[:,1], '-o', label = r'SNR= {} dB '.format(snr[1]) )
plt.plot(spectral_w, std_sw_NN[:,2], '-p', label = r'SNR= {} dB '.format(snr[2]) )
plt.grid()
plt.xlabel(r'$\sigma_p$ [m/s]')
plt.ylabel(r'Std $\hat{\sigma}_p$ [m/s]')
plt.legend()
graph.show()



             
            