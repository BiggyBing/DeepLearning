#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 01:34:40 2018

@author: bingyangwen
"""

from Generator_function import generator,sampler,label_feature_split
import numpy as np

#binary_bpsk = [decimal_to_binary(i,12) for i in range(2**12)]   
#
#BPSK = generator (binary_bpsk, 'BPSK', noise = True)
#QPSK = generator (binary_bpsk, 'QPSK', noise = True) 
#_16PSK = generator (binary_bpsk, '16PSK', noise = True)  
#
#B_f , a = label_feature_split(BPSK)
#Q_f , b = label_feature_split(QPSK)
#S_f , c = label_feature_split(_16PSK)

B = sampler(B_f,12)
Q = sampler(Q_f,12)
S = sampler(S_f,12)

#%%
x = np.linspace(0,3,12)
BPSK, = plt.plot(x, S[2835])



plt.title('16PSK sample of 101100010011')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True, which='both')
plt.axhline(y=0, color='b')