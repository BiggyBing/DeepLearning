#create distribution dataset
#normal, uniform, exponential
import numpy as np
import random
from random import randint

def dataset(datasize = 784):
    #arr = np.empty((0,784), float)
    arr=[[]]
    #Normal distribution
    for _ in range(1000):
        mean = randint(0,100)
        sd = random.uniform(0,20)
        normal_data = np.random.normal(mean,sd,datasize)
        normal_data = np.append(normal_data,0)
        normal_data.tolist()
        arr.append(normal_data)
        
        
    #uniform
    for _ in range(1000):
        low = random.uniform(0.1,100)
        variance = low/20
        high = low + random.uniform(0.001,variance)
        uniform_data = np.random.normal(low,high,datasize)
        uniform_data = np.append(uniform_data,1)
        uniform_data.tolist()
        arr.append(uniform_data)
        
    
    #exponential
    for _ in range(1000):
        scale = randint(1,100)
        ex_data = np.random.exponential(scale,datasize)
        ex_data = np.append(ex_data,2)
        ex_data.tolist()
        arr.append(ex_data)
    
    arr = arr[1:3001]
    random.shuffle(arr)
    train_set = np.asarray(arr[0:2500])
    test_set = np.asarray(arr[2501:3000])        
    
    return train_set, test_set
    
        
    
        
        