#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 13:29:35 2021

@author: Arturo Collado Rosell
email: arturo.collado.rosell@gmail.com
"""


import numpy as np
import numpy.matlib


def synthetic_IQ_data( **kwargs ):
    
    """ 
    Possible Inputs:
    clutter_power,
    clutter_s_w, 
    phenom_power, 
    phenom_vm,
    phenom_s_w,
    noise_power, 
    M, 
    wavelength,
    PRF,
    radar_mode 
    int_stagg 
    samples_factor
    num_realizations
    
    """
    # some default parameters if not provided 
    if  'M' not in kwargs:
        M = 64     #default M value
    elif kwargs['M']%2 !=0:
        M = kwargs['M'] + 1 # if M is not an even number we add 1 to convert it to an even number 
            
    if 'radar_mode' not in kwargs:
        radar_mode = 'uniform'
    
    if 'radar_mode' in kwargs:
        radar_mode = kwargs['radar_mode']
        if radar_mode == 'staggered':
            if 'int_stagg' not in kwargs:
                int_stagg = [2, 3]
    else:
        radar_mode = 'uniform'    
    
    if radar_mode == 'uniform':
        sample_index = np.arange(M)
        num_samples_uniform = M 
    else:
        num_samples_uniform = round(M - 1)* sum(int_stagg)/len(int_stagg) + 1
        sample_index = np.cumsum(np.matlib.repmat(int_stagg, 1, round(M/len(int_stagg))))
        sample_index = sample_index[:M]
    
    if 'samples_factor' not in kwargs:
        samples_factor = 10
        num_samples = samples_factor * num_samples_uniform
    
    if 'PRF' not in kwargs:
        raise Exception('Introduce a PRF value')
    else:
        PRF = kwargs['PRF']
    
    if 'wavelength' not in kwargs:
        wavelength = 3e8/5.6e9  
    else:
        wavelength = kwargs['wavelength']
    
          
    #data generation
    phenomFlag = False
    if 'phenom_power' in kwargs:
        phenomFlag = True
        phenom_power = kwargs['phenom_power']
        try:
            phenom_vm = kwargs['phenom_vm']
            phenom_s_w = kwargs['phenom_s_w']
            # converting the values to frequency space
            p_fm = - 2.0/wavelength * phenom_vm
            p_s_w_f = 2.0/wavelength * phenom_s_w
        except Exception as e:
            print(f"Exception: key{e} is not present as input")
            
        
    
    clutterFlag = False
    if 'clutter_power' in kwargs:
        clutterFlag = True
        clutter_power = kwargs['clutter_power']
        try:
            clutter_s_w = kwargs['clutter_s_w'] 
            # converting the values to frequency space
            c_s_w_f = 2.0/wavelength * clutter_s_w
        except Exception as e:
            print(f"Exception: key{e} is not present as input")
        
        
        
            
    noiseFlag = False
    if 'noise_power' in kwargs:
        noiseFlag = True
        noise_power = kwargs['noise_power']
    else:
        noise_power = 0

    if phenomFlag == False and clutterFlag == False and noiseFlag == False:
        print('There are not output data because you did not provide the necessary parameters inputs')

    #Frequency grid
    f = np.arange(-num_samples/2, num_samples/2) * PRF/num_samples      
    
    dep_p = np.zeros(num_samples,)
    if phenomFlag:
        idx_0 = round(p_fm * num_samples/PRF)
        freq_0 = p_fm - idx_0 * PRF/num_samples
        dep_p = phenom_power / np.sqrt(2 * np.pi)/ p_s_w_f * np.exp(-(f - freq_0)**2 / (2 * p_s_w_f**2))
        dep_p = np.roll(dep_p, idx_0)
        
    dep_c = np.zeros(num_samples,)
    if clutterFlag:
        
        dep_c = clutter_power / np.sqrt(2 * np.pi)/ c_s_w_f * np.exp(-(f )**2 / (2 * c_s_w_f**2))
        np.roll(dep_c, idx_0)    
        
    dep = dep_p + dep_c
    
    dep1 =  np.concatenate((dep[int(num_samples/2 + 1):], dep[0:int(num_samples/2 + 1)]))   
    
    lambdak = num_samples* PRF * dep1
    
    #Data generation using "Simulation of Weatherlike Doppler Spectra and Signals" paper
    if 'num_realizations' not in kwargs:
        num_realizations = 1
    else:
        num_realizations = kwargs['num_realizations']
    # Uniform distributed variable in (0, 1)
    U = np.random.rand(num_realizations,num_samples)
    # random variable with exponential distribution
    Pk = -1.0* np.matlib.repmat(lambdak, num_realizations, 1) * np.log(1 - U)
    
    #random phase to generate I and Q samples
    theta = 2 * np.pi * np.random.rand(num_realizations, num_samples)
    
    #Complex signal
    z_IQ =  np.fft.ifft(np.sqrt(Pk) * np.exp(1j * theta), axis = 1) 
    
    
    # truncating the output and adding the noise    
    z_IQ = z_IQ[:,sample_index] + np.sqrt(noise_power/2) * ( np.random.randn(num_realizations, M) + 1j* np.random.randn(num_realizations, M))   
    
    time = np.arange(num_samples)/PRF
    time = time[sample_index]
    
    return z_IQ, time


def DEP_estimation(z_IQ, PRF, w):
    U = sum(w**2)
    (I,M) = z_IQ.shape
    dep = np.zeros(z_IQ.shape)
    if I == 1:
        dep = 1/PRF/U * np.abs(np.fft.fft(z_IQ, axis = 1))**2
    else:
        dep = 1/PRF/U * np.sum(np.abs(np.fft.fft(z_IQ, axis = 1))**2, axis = 0 )
    return dep    
    
    

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
