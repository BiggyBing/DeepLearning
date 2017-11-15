#create distribution dataset
#normal, uniform, exponential
import numpy as np
import random
from random import randint

def dataset(datasize = 1000, data_num = 10000, Normalization = False):
    arr=[]
    #Normal distribution
    for _ in range(data_num):
        mean = randint(0,100)
        sd = random.uniform(0,20)
        normal_data = np.random.normal(mean, sd, datasize)
        if Normalization:
            normal_data = normal_data/max(normal_data)
        normal_data.tolist()
        arr.append(normal_data)
        
        
    #uniform
    for _ in range(data_num):
        low = random.uniform(0.1,100)
        variance = low/20
        high = low + random.uniform(0.001,variance)
        uniform_data = np.random.normal(low, high, datasize)
        if Normalization:
            uniform_data = uniform_data/max(uniform_data)
        uniform_data.tolist()
        arr.append(uniform_data)
        
    
    #exponential
    for _ in range(data_num):
        scale = randint(1,100)
        ex_data = np.random.exponential(scale, datasize)
        if Normalization:
            ex_data = ex_data/max(ex_data)
        ex_data.tolist()
        arr.append(ex_data)
    
    
    random.shuffle(arr)
    dataset = np.asarray(arr)
       
    
    return dataset
    
        
    
        
        