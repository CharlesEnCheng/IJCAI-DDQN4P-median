#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:20:37 2021

@author: en-chengchang
"""

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  

import os
import random
import numpy as np
from config import conf

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  

"""
n, k, c, d, mode are parameters described in Experiment
n and k are given at the entry point.
c, d, mode are given in conf.py
"""
def get_instance(n, k, c, d, mode, seed = None):

    if seed == None: pass
    else: 
        np.random.seed(seed=seed)
        random.seed(seed)
        
    # random pick k nodes for the set F denoted as f_nodes
    # F' as left_nodes
    instance = []
    tmp_set = np.random.permutation(range(n))
    f_nodes = random.sample(range(n), k)
    left_nodes = list(set(range(n)).difference(set(f_nodes)))

    # F deliver to C with random weights between [20,50]
    groups = []
    for i in range(len(f_nodes)-1):
        groups.append(tmp_set[(i)*int(np.floor(n/k)):(i+1)*int(np.floor(n/k))])
    groups.append(tmp_set[(len(f_nodes)-1)*int(np.floor(n/k)):])
    
    for s in range(len(f_nodes)):
        for node in groups[s]:
            instance.append([f_nodes[s], node, np.random.randint(20,50)])
    
    # F' deliver to C with random weights between [c,d] and two settings of degrees   
    for s in left_nodes:
        if mode == 0:
            c_nodes = random.sample(range(n), np.random.randint(1, n))
        elif mode == 1:
            c_nodes = random.sample(range(n), np.random.randint(int(n/k), int(2*n/k)))
        for c_node in c_nodes:
            instance.append([s, c_node, np.random.randint(c,d)])
 
    return sorted(instance)

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== # 

if __name__ == '__main__':
    if not os.path.isdir(conf.path.data): os.mkdir(conf.path.data)  
    
    for folder in ['train/', 'valid/', 'test/']:
        if not os.path.isdir(conf.path.data + folder): 
            os.mkdir(conf.path.data + folder)        


    for c,d,mode in [(100,400,0), (10,30,0),(20,50,1)]:
        
        # create training instances
        for figure in range(20000):
            np.random.seed(seed=figure)
            n = np.random.randint(7, 9)
            k = np.random.randint(2, n-1)
            instance = get_instance(n, k, c, d, mode, seed = figure)
            instance = np.array(instance)
            # file name c_d_mode_figure_n_k.npy
            path = conf.path.data + 'train/' + '{0:03}_{1:03}_{2}_{3:04}_{4}_{5}.npy'.format(c,d,mode,figure,n,k)
            np.save(path, instance)
            
        
        # create test & validation instances
        val_num = 100
        for figure in range(val_num):
            np.random.seed(seed=figure)
            n = 15; k = 5
            instance = get_instance(n, k, c, d, mode, seed = figure)
            instance = np.array(instance)
            path = conf.path.data + 'valid/' + '{0:03}_{1:03}_{2}_{3:04}_{4}_{5}.npy'.format(c,d,mode,figure,n,k)
            np.save(path, instance)
            
            instance = get_instance(n, k, c, d, mode, seed = figure+val_num*2)
            instance = np.array(instance)
            path = conf.path.data + 'test/' + '{0:03}_{1:03}_{2}_{3:04}_{4}_{5}_test.npy'.format(c,d,mode,figure,n,k)
            np.save(path, instance)
            
