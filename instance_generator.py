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

def get_instance(n, k, c, d, mode, seed = None):

    if seed == None: pass
    else: 
        np.random.seed(seed=seed)
        random.seed(seed)
        
    instance = []
    tmp_set = np.random.permutation(range(n))
    f_nodes = random.sample(range(n), k)
    left_nodes = list(set(range(n)).difference(set(f_nodes)))

    groups = []
    for i in range(len(f_nodes)-1):
        groups.append(tmp_set[(i)*int(np.floor(n/k)):(i+1)*int(np.floor(n/k))])
    groups.append(tmp_set[(len(f_nodes)-1)*int(np.floor(n/k)):])
    
    for s in range(len(f_nodes)):
        for node in groups[s]:
            instance.append([f_nodes[s], node, np.random.randint(20,50)])

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

    # list contents (n, k, c, d, mode, seed)
    
    # train section - n(5,9), k(2,n-1), cd=(100,400), mode = 0 
    # train section - n(5,9), k(2,n-1), cd=(10,30),   mode = 0     
    # train section - n(5,9), k(2,n-1), cd=(20,50),   mode = 1      

    for c,d,mode in [(100,400,0), (10,30,0),(20,50,1)]:
        for figure in range(20000):
            np.random.seed(seed=figure)
            n = np.random.randint(7, 9)
            k = np.random.randint(2, n-1)
            instance = get_instance(n, k, c, d, mode, seed = figure)
            instance = np.array(instance)
            # file name c_d_mode_figure_n_k.npy
            path = conf.path.data + 'train/' + '{0:03}_{1:03}_{2}_{3:04}_{4}_{5}.npy'.format(c,d,mode,figure,n,k)
            np.save(path, instance)
            
    # valid & test section - n(15), k(5), cd=(100,400), mode = 0 
    # valid & test section - n(15), k(5), cd=(10,30),   mode = 0     
    # valid & test section - n(15), k(5), cd=(20,50),   mode = 1    
         
        val_num = 100
        for figure in range(val_num):
            np.random.seed(seed=figure)
            n = 15; k = 5
            instance = get_instance(n, k, c, d, mode, seed = figure)
            instance = np.array(instance)
            # file name c_d_mode_figure.txt
            path = conf.path.data + 'valid/' + '{0:03}_{1:03}_{2}_{3:04}_{4}_{5}.npy'.format(c,d,mode,figure,n,k)
            np.save(path, instance)
            
            instance = get_instance(n, k, c, d, mode, seed = figure+val_num*2)
            instance = np.array(instance)
            # file name c_d_mode_figure.txt
            path = conf.path.data + 'test/' + '{0:03}_{1:03}_{2}_{3:04}_{4}_{5}_test.npy'.format(c,d,mode,figure,n,k)
            np.save(path, instance)
            
