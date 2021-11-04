#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 16:14:59 2021

@author: Arturo Collado Rosell
This script is to reproduce one of the experiment of the original paper, the one varying 
the clutter width
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

np.random.seed(2021) # seed for reproducibility 

Tu = 0.25e-3 # If sataggered Tu  = T2 - T1, else Tu is the PRI [s]
PRF = 1/Tu #[Hz]
M = 64 #Samples number in a CPI 
Fc = 5.6e9# Carrier frequency [Hz]
c = 3.0e8 # speed of ligth [m/s]
theta_3dB_acimut = 1 # half power angle, acimut direction [degrees]
wavelenght = c/Fc # [m]
va = wavelenght*PRF/4 # maximum unambigous velocity [m/s]
Sp = 1 # phenomenon power [A.U]
vm = [0.2*va, 0.4*va] #Phenomenon Doppler velocities for simulation [m/s]
spectral_w = [3, 5] # Phenomenon spectral width for simulation [m/s]
csr = 40 # Clutter to signal ratio [dB]
Sc = Sp * 10**(csr/10)

radar_mode = 'staggered'
int_stagg = [2, 3]

snr = 20
power_noise = Sp / (10**(snr/10))

I = 1

w =  theta_3dB_acimut * np.pi/180.0 / (M * sum(int_stagg)/len(int_stagg)* Tu)
theoretical_clutter_spectral_width = w*wavelenght * np.sqrt(np.log(2.0)) / (2.0 * np.pi * theta_3dB_acimut* np.pi/180.0) # [m/s]

clutter_width = np.linspace(0.1, 0.6, 20)

parameters_with_clutter = {      'clutter_power':Sc,       
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
num_samples_uniform = round((M - 1)* sum(int_stagg)/len(int_stagg)) + 1
window = np.kaiser(num_samples_uniform, 8)



###########Data generation#####################################
L = 1000 # Monte Carlo realization number
N_vel = len(vm)
N_sw_c = len(clutter_width)
N_sw = len(spectral_w) 
complex_IQ_data = np.zeros((N_vel*N_sw_c*N_sw*L, M), dtype = complex)
data_PSD = np.zeros((N_vel*N_sw_c*N_sw*L, num_samples_uniform))
for i in synthetic_weather_data_IQ.progressbar(range(N_vel), 'Computing:') :
    for q in range(N_sw_c):
        for j in range(N_sw):
            parameters_with_clutter['phenom_vm'] = vm[i]
            parameters_with_clutter['clutter_s_w'] = clutter_width[q]
            parameters_with_clutter['phenom_s_w'] = spectral_w[j]
            for ind_l in range(L):
                z_IQ, _ = synthetic_weather_data_IQ.synthetic_IQ_data(**parameters_with_clutter)
                complex_IQ_data[i*N_sw_c*N_sw*L + q*N_sw*L + j*L + ind_l,:] = z_IQ        
                if radar_mode == 'staggered':
                    z_IQ_zero_interp = synthetic_weather_data_IQ.zero_interpolation(z_IQ, int_stagg)
                    data_w = z_IQ_zero_interp * window
                else:
                    data_w = z_IQ * window
                        
                ##PSD estimation
                psd = synthetic_weather_data_IQ.PSD_estimation(data_w, w = window, PRF = 1/Tu)
                psd = psd /np.max(psd)
                data_PSD[i*N_sw_c*N_sw*L + q*N_sw*L + j*L + ind_l,:] = 10*np.log10(psd[0,:])

###########################Estimations###########################################                
clutter_power = synthetic_weather_data_IQ.clutter_power(complex_IQ_data, 2*Tu, 3*Tu, theoretical_clutter_spectral_width * 2/wavelenght, window = 'Kaiser', alpha = 8)                
#predictions using the NN
model = tf.keras.models.load_model('GPU_100_512model.h5')
vel_pred, sw_pred, csr_pred = RadarNet.prediction(model, data_PSD, device = '/GPU:0')                

try:
    meta_params = np.load('some_params_to_train.npy')
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

vel_NN = np.zeros((N_sw, N_sw_c, N_vel))
std_vel_NN = np.zeros((N_sw, N_sw_c, N_vel))
sw_NN = np.zeros((N_sw, N_sw_c, N_vel))
std_sw_NN = np.zeros((N_sw, N_sw_c, N_vel))
pow_NN = np.zeros((N_sw, N_sw_c, N_vel))
std_pow_NN = np.zeros((N_sw, N_sw_c, N_vel))
   
for i in range(N_vel):
    for q in range(N_sw_c):
        for j in range(N_sw):
            batch_vel = vel_pred[i*N_sw_c*N_sw*L + q*N_sw*L + j*L: i*N_sw_c*N_sw*L + q*N_sw*L + (j+1)*L]
            batch_sw = sw_pred[i*N_sw_c*N_sw*L + q*N_sw*L + j*L: i*N_sw_c*N_sw*L + q*N_sw*L + (j+1)*L]
            batch_csr = csr_pred[i*N_sw_c*N_sw*L + q*N_sw*L + j*L: i*N_sw_c*N_sw*L + q*N_sw*L + (j+1)*L] 
            
            clutt_power_batch = clutter_power[i*N_sw_c*N_sw*L + q*N_sw*L + j*L: i*N_sw_c*N_sw*L + q*N_sw*L + (j+1)*L] 
            power_phenom_batch = clutt_power_batch / (10.0**(batch_csr/10.0)) * (batch_csr >= 0 )
            
            #Bias and standard deviation
            pow_NN[j,q,i], std_pow_NN[j,q,i] = np.mean(power_phenom_batch), np.std(power_phenom_batch)           
            vel_NN[j,q,i], std_vel_NN[j,q,i] = np.mean(batch_vel), np.std(batch_vel)             
            sw_NN[j,q,i], std_sw_NN[j,q,i] = np.mean(batch_sw), np.std(batch_sw)
             
            
###Ploting the results##############################################################

#Power Bias
graph = plt.figure(figsize = (7,4))
plt.plot(clutter_width, pow_NN[0,:,0] - Sp*np.ones((N_sw_c,)),'-.', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[0]))
plt.plot(clutter_width, pow_NN[0,:,1] - Sp*np.ones((N_sw_c,)), '-o' , label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[0]))
plt.plot(clutter_width, pow_NN[1,:,0] - Sp*np.ones((N_sw_c,)), '-p', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[1]))
plt.plot(clutter_width, pow_NN[1,:,1] - Sp*np.ones((N_sw_c,)), '-x' , label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[1]))
plt.grid()
plt.xlabel(r'$\sigma_c$ [m/s]')
plt.ylabel(r'Bias $\hat{p}_p$ [m/s]')
plt.legend()
graph.show()


#Power Std
graph = plt.figure(figsize = (7,4))
plt.plot(clutter_width, std_pow_NN[0,:,0], '-.', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[0]) )
plt.plot(clutter_width, std_pow_NN[0,:,1], '-o', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[0]) )
plt.plot(clutter_width, std_pow_NN[1,:,0], '-p', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[1]) )
plt.plot(clutter_width, std_pow_NN[1,:,1], '-x', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[1]) )
plt.grid()
plt.xlabel(r'$\sigma_c$ [m/s]')
plt.ylabel(r'Std $\hat{p}_p$ [m/s]')
plt.legend()
graph.show()


#Velocity Bias
graph = plt.figure(figsize = (7,4))
plt.plot(clutter_width, vel_NN[0,:,0] - vm[0]*np.ones((N_sw_c,)),'-.', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[0]))
plt.plot(clutter_width, vel_NN[0,:,1] - vm[1]*np.ones((N_sw_c,)), '-o' , label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[0]))
plt.plot(clutter_width, vel_NN[1,:,0] - vm[0]*np.ones((N_sw_c,)), '-p', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[1]))
plt.plot(clutter_width, vel_NN[1,:,1] - vm[1]*np.ones((N_sw_c,)), '-x' , label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[1]))
plt.grid()
plt.xlabel(r'$\sigma_c$ [m/s]')
plt.ylabel(r'Bias $\hat{v}_p$ [m/s]')
plt.legend()
graph.show()

#Velocity Std
graph = plt.figure(figsize = (7,4))
plt.plot(clutter_width, std_vel_NN[0,:,0], '-.', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[0]) )
plt.plot(clutter_width, std_vel_NN[0,:,1], '-o', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[0]) )
plt.plot(clutter_width, std_vel_NN[1,:,0], '-p', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[1]) )
plt.plot(clutter_width, std_vel_NN[1,:,1], '-x', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[1]) )
plt.grid()
plt.xlabel(r'$\sigma_c$ [m/s]')
plt.ylabel(r'Std $\hat{v}_p$ [m/s]')
plt.legend()
graph.show()
            

#spectrum width Bias
graph = plt.figure(figsize = (7,4))
plt.plot(clutter_width, sw_NN[0,:,0] - spectral_w[0]*np.ones((N_sw_c,)), '-.', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[0]) )
plt.plot(clutter_width, sw_NN[0,:,1] - spectral_w[0]*np.ones((N_sw_c,)), '-o', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[0]) )
plt.plot(clutter_width, sw_NN[1,:,0] - spectral_w[1]*np.ones((N_sw_c,)), '-p', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[1]) )
plt.plot(clutter_width, sw_NN[1,:,1] - spectral_w[1]*np.ones((N_sw_c,)), '-x', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[1]) )
plt.grid()
plt.xlabel(r'$\sigma_c$ [m/s]')
plt.ylabel(r'Bias $\hat{\sigma}_p$ [m/s]')
plt.legend()
graph.show()

#spectrum width Std
graph = plt.figure(figsize = (7,4))
plt.plot(clutter_width, std_sw_NN[0,:,0], '-.', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[0]) )
plt.plot(clutter_width, std_sw_NN[0,:,1], '-o', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[0]) )
plt.plot(clutter_width, std_sw_NN[1,:,0], '-p', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[0]/va, spectral_w[1]) )
plt.plot(clutter_width, std_sw_NN[1,:,1], '-x', label = r'$v_p = {:.3} v_a, \sigma_p = {}$ m/s '.format(vm[1]/va, spectral_w[1]) )
plt.grid()
plt.xlabel(r'$\sigma_c$ [m/s]')
plt.ylabel(r'Std $\hat{\sigma}_p$ [m/s]')
plt.legend()
graph.show()

    
                
    
    
