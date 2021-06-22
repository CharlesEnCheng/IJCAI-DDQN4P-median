#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:27:47 2021

@author: en-chengchang
"""

class conf:

    class general:
        c = 100
        d = 400
        mode = 0
        
        seed = 42

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
        xpress = '/Users/en-chengchang/Desktop/FICO/'
        
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

 
