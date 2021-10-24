#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 17:24:33 2021

@author: Arturo Collado Rosell

This script is for reproduce one of the experiment of the original paper, the one varying 
the CSR

"""
import numpy as np
import synthetic_weather_data_IQ

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
csr = np.linspace(0, 50, 10)
Sc = Sp * 10**(csr/10)

radar_mode = 'staggered'
int_stagg = [2, 3]

snr = 20
power_noise = Sp / (10**(snr/10))

I = 1

w =  theta_3dB_acimut * np.pi/180.0 / (M * sum(int_stagg)/len(int_stagg)* Tu)
clutter_spectral_width = w*wavelenght * np.sqrt(np.log(2.0)) / (2.0 * np.pi * theta_3dB_acimut* np.pi/180.0) # [m/s]

# clutter_power: power of clutter [times]
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
                
                
                
                
    
