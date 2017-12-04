# -*- coding: utf-8 -*-
import numpy as np
import random

def BPSK_QPSK_Generator(Bit_size, Num_shift = 1, Sample_Noise = False, Disturbance =False, Shift = False):
    '''
    Bit_size : Number of bits within one binary code
    Num_shift: Num_shift denote the number of modulated signals you want to create for each binary code. 
               For each binary code, the modulated signal may vary with carriar signal that have different offsets. 
    Shift： Shift denote if set the offset of carriar signal or not. If it is set False, Num_shift should be 1.
    Sample_Noise: In our case, the period of carriar signal is 2pi and sample frequency is pi/2. As a result we could get four sample points within one cycle
                  Sample_Noise represents the variantion when sample the points, which makes sample frequency variate around pi/2.
    Disturbance : It is another form of noise, which represent the a 'bad' sine carriar wave
    
    
    '''
    
    def decimal_to_binary(num, _bit_size):
        temp = bin(num)
        temp = temp[2:]
        
        if len(temp) !=_bit_size:
            diff = _bit_size - len(temp)
            a = '0'
            for _ in range(diff-1):
                a = a + '0'
            return a+ temp
        else:
            return temp
    
    phase_shift = np.array([np.pi*i/3 for i in range(8)])
        
    def EPSK_Generator(Binary_data, _bit_size, offset=0, S_N = Sample_Noise, D = Disturbance):
        
        if S_N == True:
            noise = random.uniform(-np.pi/12,np.pi/12)
        else:
            noise = 0
            
        if D == True:
            disturbance = random.gauss(0,0.3)
        else:
            disturbance = 0
            
        
        phase_shift = np.array([np.pi*i/4 for i in range(8)])
        sample = np.array([np.pi*i/2 for i in range(12)])
        time_domain_data = np.array([])
        i = 0
        while (i < _bit_size-1):
            temp = Binary_data[i]+Binary_data[i+1]+Binary_data[i+2]
            mark = int(temp,2)
            temp_modulation = np.array([np.sin(phase_shift[mark]+offset+noise+sample[j]) + disturbance for j in range(12)])
            time_domain_data = np.append(time_domain_data,temp_modulation)
            i = i + 3
        time_domain_data = np.append(time_domain_data, np.array([0,0,1]))
        return time_domain_data#.tolist()
    
    
    def QPSK_Generator(Binary_data, _bit_size, offset=0, S_N = Sample_Noise, D = Disturbance):
        
        if S_N == True:
            noise = random.uniform(-np.pi/12,np.pi/12)
        else:
            noise = 0
            
        if D == True:
            disturbance = random.gauss(0,0.3)
        else:
            disturbance = 0
            
        i = 0
        time_domain_data = np.array([])
        while (i < _bit_size-1):
            temp = Binary_data[i]+Binary_data[i+1]
            
            if temp == '00':
                time_domain_data = np.append(time_domain_data, np.array([np.sin(offset+noise)+disturbance,np.sin(offset+np.pi/2+noise)+disturbance, 
                                                                         np.sin(offset+np.pi+noise)+disturbance,np.sin(offset+3*np.pi/2+noise)+disturbance,
                                                                         np.sin(offset+2*np.pi+noise)+disturbance,np.sin(offset+5*np.pi/2+noise)+disturbance, 
                                                                         np.sin(offset+3*np.pi+noise)+disturbance,np.sin(offset+7*np.pi/2+noise)+disturbance]))   
            elif temp == '10':
                time_domain_data = np.append(time_domain_data, np.array([np.sin(offset+np.pi+noise)+disturbance, np.sin(offset+3*np.pi/2+noise)+disturbance,
                                                                        np.sin(offset+2*np.pi+noise)+disturbance,np.sin(offset+5*np.pi/2+noise)+disturbance,
                                                                        np.sin(offset+3*np.pi+noise)+disturbance, np.sin(offset+7*np.pi/2+noise)+disturbance,
                                                                        np.sin(offset+4*np.pi+noise)+disturbance,np.sin(offset+9*np.pi/2+noise)+disturbance]))
            elif temp == '01':
                time_domain_data = np.append(time_domain_data, np.array([np.sin(offset+np.pi/2+noise)+disturbance,np.sin(offset+np.pi+noise)+disturbance, 
                                                                         np.sin(offset+3*np.pi/2+noise)+disturbance,np.sin(offset+2*np.pi+noise)+disturbance,
                                                                         np.sin(offset+5*np.pi/2+noise)+disturbance,np.sin(offset+3*np.pi+noise)+disturbance, 
                                                                         np.sin(offset+7*np.pi/2+noise)+disturbance,np.sin(offset+4*np.pi+noise)+disturbance]))
            elif temp == '11':
                time_domain_data = np.append(time_domain_data, np.array([np.sin(offset+3*np.pi/2+noise)+disturbance,np.sin(offset+2*np.pi+noise)+disturbance,
                                                                         np.sin(offset+5*np.pi/2+noise)+disturbance,np.sin(offset+3*np.pi+noise)+disturbance,
                                                                         np.sin(offset+7*np.pi/2+noise)+disturbance,np.sin(offset+4*np.pi+noise)+disturbance,
                                                                         np.sin(offset+9*np.pi/2+noise)+disturbance,np.sin(offset+5*np.pi+noise)+disturbance]))
            i = i + 2
        time_domain_data = np.append(time_domain_data, np.array([0,1,0]))
        return time_domain_data#.tolist()
    
    def BPSK_Generator(Binary_data,offset, S_N = Sample_Noise, D = Disturbance):
        
        if S_N == True:
            noise = random.uniform(-np.pi/12,np.pi/12)
        else:
            noise = 0
            
        if D == True:
            disturbance = random.gauss(0,0.3)
        else:
            disturbance = 0
            
        i = 0
        time_domain_data = np.array([])
        for i in range(len(Binary_data)):
            if Binary_data[i] == '1':
                time_domain_data = np.append(time_domain_data, np.array([np.sin(offset+noise)+disturbance,np.sin(offset+np.pi/2+noise)+disturbance, 
                                                                         np.sin(offset+np.pi+noise)+disturbance,np.sin(offset+3*np.pi/2+noise)+disturbance]))   
            elif Binary_data[i] == '0':
                time_domain_data = np.append(time_domain_data, np.array([np.sin(offset+np.pi+noise)+disturbance,np.sin(offset+3*np.pi/2+noise)+disturbance, 
                                                                         np.sin(offset+2*np.pi+noise)+disturbance,np.sin(offset+5*np.pi/2+noise)+disturbance]))  
        time_domain_data = np.append(time_domain_data, np.array([1,0,0]))
        return time_domain_data#.tolist()
            
            
            
            
    # Create all 12-bits binary from 0(0000 0000 0000) to 4095(1111 1111 1111)      
    Binary_data = []
    for i in range(pow(2,Bit_size)):
        Binary_data.append(decimal_to_binary(i, _bit_size = Bit_size))
    random.shuffle(Binary_data)
    
    
    BPSK = []
    for i in range(len(Binary_data)):
        for j in range(Num_shift):
            if Shift:
                offset = random.uniform(0, np.pi/2)
            else:
                offset = 0
            temp = BPSK_Generator(Binary_data[i], offset)
            BPSK.append(temp)
    
    QPSK = []
    for i in range(len(Binary_data)):
        for j in range(Num_shift):
            if Shift:
                offset = random.uniform(0, np.pi/2)
            else:
                offset = 0
            temp = QPSK_Generator(Binary_data[i], Bit_size, offset)
            QPSK.append(temp)
            
    EPSK = []
    for i in range(len(Binary_data)):
        for j in range(Num_shift):
            if Shift:
                offset = random.uniform(0, np.pi/2)
            else:
                offset = 0
            temp = EPSK_Generator(Binary_data[i], Bit_size, offset)
            EPSK.append(temp)
        
    BPSK = np.array(BPSK)
    QPSK = np.array(QPSK)
    EPSK = np.array(EPSK)   
    
    return BPSK, QPSK, EPSK
