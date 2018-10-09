#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 23:44:16 2018

@author: bingyangwen
"""

import numpy as np
from random import shuffle

g_Tc = 1
g_A = 1
time = np.arange(0,g_Tc,g_Tc/50)
plot_time = np.arange(0,12*g_Tc,g_Tc/50)
plot_sample = np.arange(0,3*g_Tc,g_Tc/4)
sample_point = np.arange(0,g_Tc,g_Tc/4)

#work as a modulator that change binary to signal
def generator(DigtSeq, MPSK, noise = False, noise_v = 0.1):
    '''
    DigtSeq is a 1D binary vector in any data type e.g ['10101','01011','10101','01010']
    '''
    MPSK = MPSK.upper()
    modulation = []
    #sample = []
    
    if MPSK == "BPSK":
        for i in range(len(DigtSeq)):
            t_modulation = [MPSK]
            t_sample = []
            for j in range(len(DigtSeq[i])):
                shift = int(DigtSeq[i][j], 2)
                if noise:
                    white_noise = np.random.normal(0, noise_v)
                    t_modulation.extend(g_A * np.cos(2*np.pi/g_Tc * time + np.pi * (1 - shift)) + white_noise)
                    #t_sample.extend(g_A * np.cos(2*np.pi/g_Tc * sample_point + np.pi * (1 - shift)))
                else:
                    t_modulation.extend(g_A * np.cos(2*np.pi/g_Tc * time + np.pi * (1 - shift)))
                    #t_sample.extend(g_A * np.cos(2*np.pi/g_Tc * sample_point + np.pi * (1 - shift)))        
            modulation.append(t_modulation)
            
            #sample.append(t_sample)
        
    elif MPSK == "QPSK":
        for i in range(len(DigtSeq)):
            t_modulation = [MPSK]
            t_sample = []
            check = [3,2,4,1]
            for j in range(0,len(DigtSeq[i]),2):
                shift = int(DigtSeq[i][j:j+2], 2)
                if noise:
                    white_noise = np.random.normal(0, noise_v)
                    t_modulation.extend(g_A * np.cos(2*np.pi/g_Tc * time + np.pi * (2*check[shift]-1)/4) + white_noise)
                #t_sample.extend(g_A * np.cos(2*np.pi/g_Tc * sample_point + np.pi * (2*check[shift]-1)/4))
                else:
                    t_modulation.extend(g_A * np.cos(2*np.pi/g_Tc * time + np.pi * (2*check[shift]-1)/4))
            
            modulation.append(t_modulation)
            
        
            #sample.append(t_sample)
    elif MPSK == "16PSK":
        for i in range(len(DigtSeq)):
            t_modulation = [MPSK]
            t_sample = []
            check = [0,1,3,2,7,6,4,5,15,14,12,13,8,9,11,10]
            for j in range(0,len(DigtSeq[i]),4):
                shift = int(DigtSeq[i][j:j+4], 2)
                if noise:
                    white_noise = np.random.normal(0, noise_v)
                    t_modulation.extend(g_A * np.cos(2*np.pi/g_Tc * time + np.pi * check[shift]/8) + white_noise)
                #t_sample.extend(g_A * np.cos(2*np.pi/g_Tc * sample_point + np.pi * check[shift]/8))
                else:
                    t_modulation.extend(g_A * np.cos(2*np.pi/g_Tc * time + np.pi * check[shift]/8))
                  
            modulation.append(t_modulation)
            
            #sample.append(t_sample)
    
    elif MPSK == "64PSK":
        pass
    
    else:
        print ("Error! PSK type only allow 2, 4, 16, 64")
    return modulation#, sample

    
def sampler (dataset, output_points, sample_points = 4):
    #sample_points in another word, denote sample frequency
    #48 below decided by time segements
    sampler_base = np.linspace(0,48,sample_points+1)[0:sample_points]
    v_1 = output_points//sample_points
    v_2 = output_points%sample_points
    sampler_index = []
    sample = []
    for i in range(0,v_1):
        sampler_index.extend(sampler_base + 50 * i)
    for item in sampler_base[0:v_2]:
        sampler_index.extend([item + 50 * v_1])
    sampler_index = [int(sampler_index[i]) for i in range(len(sampler_index))]

    #return sampler_index

    for item in dataset:
        s_sample = [item[i] for i in sampler_index]
        sample.append(s_sample)
        
    return np.asarray(sample)



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
    
    
def combine_data(_BPSK, QPSK, _16PSK, feature_size):
    dataset = []
    for i in range(4096):
        dataset.append([BPSK[i][j] for j in range(0, feature_size+1)])
    for i in range(4096):
        dataset.append([QPSK[i][j] for j in range(0, feature_size+1)])
    for i in range(4096):
        dataset.append([_16PSK[i][j] for j in range(0, feature_size+1)])
    shuffle (dataset)
    
    return dataset

from sklearn.preprocessing import LabelEncoder

def label_feature_split(dataset):
    label = []
    feature = []
    for i in range(len(dataset)):
        label.append(dataset[i][0])
        feature.append(dataset[i][1:])
    feature = np.asarray(feature)
    feature = np.expand_dims(feature, axis =2)
    le = LabelEncoder()
    le.fit(label)
    label = le.transform(label)
    print (le.classes_)

    return feature, label

def combine_data(_BPSK, _QPSK, _16PSK, feature_size):
    dataset = []
    for i in range(len(_BPSK)):
        dataset.append([_BPSK[i][j] for j in range(0, feature_size+1)])
    for i in range(len(_QPSK)):
        dataset.append([_QPSK[i][j] for j in range(0, feature_size+1)])
    for i in range(len(_16PSK)):
        dataset.append([_16PSK[i][j] for j in range(0, feature_size+1)])
    #shuffle (dataset)
    
    return dataset
#%%
def prediction(model, features, labels):
    prediction = model.predict(features)
    prediction = np.round(prediction)
    labels = np.asarray(labels)
    error = 0
    error_collection = []
    for i in range(len(labels)):
        count = int(np.sum(np.absolute(labels[i] - prediction[i])))        
        if count != 0:
            error +=1                 
            worry = [i, labels[i],prediction[i]]
            error_collection.append(worry)
    
    #error_collection = error_collection.tolist()
    for item in error_collection:
        for i in range(1,len(item)):
            if item[i][0] == 1.:
                item[i] = '16PSK'
            if item[i][1] == 1.:
                item[i] = 'BPSK'
            if item[i][2] == 1.:
                item[i] = 'QPSK'
                
            
    
    accuracy = (len(labels) - error)/len(labels)

    
    return error, error_collection
#%%
def data_generator(bit_num, output_points = 12, sample = False, sample_points = 4):
    #output_points, sample_points  used for sampler()
    binary_bpsk = [decimal_to_binary(i,bit_num) for i in range(2**bit_num)]   

    train_BPSK = generator (binary_bpsk, 'BPSK', noise = True)
    train_QPSK = generator (binary_bpsk, 'QPSK', noise = True) 
    train_16PSK = generator (binary_bpsk, '16PSK', noise = True)     
    
    test_BPSK = generator (binary_bpsk, 'BPSK', noise = True)
    test_QPSK = generator (binary_bpsk, 'QPSK', noise = True) 
    test_16PSK = generator (binary_bpsk, '16PSK', noise = True)     
    
    train_data = combine_data(train_BPSK,train_QPSK,train_16PSK,150)
    test_data = combine_data(test_BPSK,test_QPSK,test_16PSK,150)
    
    train_feature, train_label = label_feature_split(train_data)
    test_feature, test_label = label_feature_split(test_data)
    
    if sample:
        train_feature = sampler(train_feature, output_points, sample_points)
        test_feature = sampler(test_feature, output_points, sample_points)
    
    return train_feature, train_label, test_feature, test_label