#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:27:47 2021

@author: en-chengchang
"""

"""
feature = True; gnn = False ==> feature based
feature = False; gnn = True ==> fixed gnn
feature & gnn = True; concat = True ==> two feature sets
feature & gnn = True; concat = False ==> embed on features
"""

"""
best_memory_play = memory set for recording play with the best obj 
"""

class conf:

    class general:
        c = 20
        d = 50
        mode = 1
        
        seed = 42
        
        feature = False
        gnn = True
        concat = False
        
        cur_epi = 1
        min_epi = 0.1
        decay_epi = 0.999
        penalty = -10**4
        
        n_replay = 100
        num_features = 10
    
    class path:
        data = './instance/'
        log  = './log/'
        model = './model/'
        
    class rl:
        gamma = 1
    
    class gnn:
        T = 5
        p_dim = 64
        
    class memory:
        collect_steps = 3000
        max_cap = 10**5
        lr = 0.0001
        batch = 64
        best_memory_play = False
        
        
    