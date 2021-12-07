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
#window
num_samples_uniform = round((M - 1)* sum(int_stagg)/len(int_stagg)) 
window = np.kaiser(num_samples_uniform, 8)


###########Data generation#####################################
L = 1000 # Monte Carlo realization number
N_vel = len(vm)
N_Sc = len(Sc)
N_sw = len(spectral_w) 
complex_IQ_data = np.zeros((N_vel*N_Sc*N_sw*L, M), dtype = complex)
data_PSD = np.zeros((N_vel*N_Sc*N_sw*L, num_samples_uniform))
for i in synthetic_weather_data_IQ.progressbar(range(N_vel), 'Computing:') :
    for q in range(N_Sc):
        for j in range(N_sw):
            parameters_with_clutter['phenom_vm'] = vm[i]
            parameters_with_clutter['clutter_power'] = Sc[q]
            parameters_with_clutter['phenom_s_w'] = spectral_w[j]
            for ind_l in range(L):
                z_IQ, _ = synthetic_weather_data_IQ.synthetic_IQ_data(**parameters_with_clutter)
                complex_IQ_data[i*N_Sc*N_sw*L + q*N_sw*L + j*L + ind_l,:] = z_IQ        
                if radar_mode == 'staggered':
                    z_IQ_zero_interp = synthetic_weather_data_IQ.zero_interpolation(z_IQ, int_stagg)
                    data_w = z_IQ_zero_interp * window
                else:
                    data_w = z_IQ * window
                        
                ##PSD estimation
                psd = synthetic_weather_data_IQ.PSD_estimation(data_w, w = window, PRF = 1/Tu)
                psd = psd /np.max(psd)
                data_PSD[i*N_Sc*N_sw*L + q*N_sw*L + j*L + ind_l,:] = 10*np.log10(psd[0,:])
                
###########################Estimations########################################## 
################Clutter power###################################################           
clutter_power = synthetic_weather_data_IQ.clutter_power(complex_IQ_data, 2*Tu, 3*Tu, clutter_spectral_width * 2/wavelenght, window = 'Kaiser', alpha = 8)                
################################################################################
#predictions using the NN

model = tf.keras.models.load_model('../plot_training/'+'GPU_100_512model.h5')
vel_pred, sw_pred, csr_pred, time = RadarNet.prediction(model, data_PSD, device = '/CPU:0')                

try:
    meta_params = np.load('../training_data/some_params_to_train.npy')
    N_vel_grid = int(meta_params[0])
    N_s_w_grid = int(meta_params[1])
    N_csr_grid = int(meta_params[2])
except Exception as e:
    print(e)  
    
# CSR grid
csr_grid = np.linspace(0, 50, N_csr_grid-1)
csr_grid = np.concatenate(([-100], csr_grid))
# spectral width grid
s_width_grid = np.linspace(0.04, 0.2, N_s_w_grid) * va
# velocity grid
vel_step = 2.0/N_vel_grid * va
vel_grid = -np.arange(-va + vel_step/2.0, va , vel_step)    


vel_pred = vel_grid[vel_pred]
sw_pred = s_width_grid[sw_pred] 
csr_pred = csr_grid[csr_pred]

vel_NN = np.zeros((N_sw, N_Sc, N_vel))
std_vel_NN = np.zeros((N_sw, N_Sc, N_vel))
sw_NN = np.zeros((N_sw, N_Sc, N_vel))
std_sw_NN = np.zeros((N_sw, N_Sc, N_vel))
pow_NN = np.zeros((N_sw, N_Sc, N_vel))
std_pow_NN = np.zeros((N_sw, N_Sc, N_vel))
   
for i in range(N_vel):
    for q in range(N_Sc):
        for j in range(N_sw):
            batch_vel = vel_pred[i*N_Sc*N_sw*L + q*N_sw*L + j*L: i*N_Sc*N_sw*L + q*N_sw*L + (j+1)*L]
            batch_sw = sw_pred[i*N_Sc*N_sw*L + q*N_sw*L + j*L: i*N_Sc*N_sw*L + q*N_sw*L + (j+1)*L]
            batch_csr = csr_pred[i*N_Sc*N_sw*L + q*N_sw*L + j*L: i*N_Sc*N_sw*L + q*N_sw*L + (j+1)*L] 
            
            clutt_power_batch = clutter_power[i*N_Sc*N_sw*L + q*N_sw*L + j*L: i*N_Sc*N_sw*L + q*N_sw*L + (j+1)*L] 
            power_phenom_batch = clutt_power_batch / (10.0**(batch_csr/10.0)) * (batch_csr >= 0 )
            
            #Bias and standard deviation
            pow_NN[j,q,i], std_pow_NN[j,q,i] = np.mean(power_phenom_batch), np.std(power_phenom_batch)           
            vel_NN[j,q,i], std_vel_NN[j,q,i] = np.mean(batch_vel), np.std(batch_vel)             
            sw_NN[j,q,i], std_sw_NN[j,q,i] = np.mean(batch_sw), np.std(batch_sw)
             
            
###Ploting the results##############################################################

#Power Bias
graph = plt.figure(figsize = (7,4))
plt.plot(csr, pow_NN[0,:,0] - Sp*np.ones((N_Sc,)),'-.', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[0]))
plt.plot(csr, pow_NN[0,:,1] - Sp*np.ones((N_Sc,)), '-o' , label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[0]))
plt.plot(csr, pow_NN[1,:,0] - Sp*np.ones((N_Sc,)), '-p', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[1]))
plt.plot(csr, pow_NN[1,:,1] - Sp*np.ones((N_Sc,)), '-x' , label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[1]))
plt.grid()
plt.xlabel('CSR [dB]')
plt.ylabel(r'Bias $\hat{p}_p$ [m/s]')
plt.legend()
graph.show()


#Power Std
graph = plt.figure(figsize = (7,4))
plt.plot(csr, std_pow_NN[0,:,0], '-.', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[0]) )
plt.plot(csr, std_pow_NN[0,:,1], '-o', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[0]) )
plt.plot(csr, std_pow_NN[1,:,0], '-p', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[1]) )
plt.plot(csr, std_pow_NN[1,:,1], '-x', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[1]) )
plt.grid()
plt.xlabel('CSR [dB]')
plt.ylabel(r'Std $\hat{p}_p$ [m/s]')
plt.legend()
graph.show()


#Velocity Bias
graph = plt.figure(figsize = (7,4))
plt.plot(csr, vel_NN[0,:,0] - vm[0]*np.ones((N_Sc,)),'-.', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[0]))
plt.plot(csr, vel_NN[0,:,1] - vm[1]*np.ones((N_Sc,)), '-o' , label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[0]))
plt.plot(csr, vel_NN[1,:,0] - vm[0]*np.ones((N_Sc,)), '-p', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[1]))
plt.plot(csr, vel_NN[1,:,1] - vm[1]*np.ones((N_Sc,)), '-x' , label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[1]))
plt.grid()
plt.xlabel('CSR [dB]')
plt.ylabel(r'Bias $\hat{v}_p$ [m/s]')
plt.legend()
graph.show()

#Velocity Std
graph = plt.figure(figsize = (7,4))
plt.plot(csr, std_vel_NN[0,:,0], '-.', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[0]) )
plt.plot(csr, std_vel_NN[0,:,1], '-o', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[0]) )
plt.plot(csr, std_vel_NN[1,:,0], '-p', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[1]) )
plt.plot(csr, std_vel_NN[1,:,1], '-x', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[1]) )
plt.grid()
plt.xlabel('CSR [dB]')
plt.ylabel(r'Std $\hat{v}_p$ [m/s]')
plt.legend()
graph.show()
            

#spectrum width Bias
graph = plt.figure(figsize = (7,4))
plt.plot(csr, sw_NN[0,:,0] - spectral_w[0]*np.ones((N_Sc,)), '-.', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[0]) )
plt.plot(csr, sw_NN[0,:,1] - spectral_w[0]*np.ones((N_Sc,)), '-o', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[0]) )
plt.plot(csr, sw_NN[1,:,0] - spectral_w[1]*np.ones((N_Sc,)), '-p', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[1]) )
plt.plot(csr, sw_NN[1,:,1] - spectral_w[1]*np.ones((N_Sc,)), '-x', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[1]) )
plt.grid()
plt.xlabel('CSR [dB]')
plt.ylabel(r'Bias $\hat{\sigma}_p$ [m/s]')
plt.legend()
graph.show()

#spectrum width Std
graph = plt.figure(figsize = (7,4))
plt.plot(csr, std_sw_NN[0,:,0], '-.', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[0]) )
plt.plot(csr, std_sw_NN[0,:,1], '-o', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[0]) )
plt.plot(csr, std_sw_NN[1,:,0], '-p', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[1]) )
plt.plot(csr, std_sw_NN[1,:,1], '-x', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[1]) )
plt.grid()
plt.xlabel('CSR [dB]')
plt.ylabel(r'Std $\hat{\sigma}_p$ [m/s]')
plt.legend()
graph.show()

