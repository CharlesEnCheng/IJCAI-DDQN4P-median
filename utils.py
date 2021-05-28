#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 19:40:52 2021

@author: en-chengchang
"""

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  

import os, datetime
import torch
from config import conf

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  

class NodeValues:
    
    def __init__(self, n, k, embed_model, value_model, adj_mat, edge_mat):
        self.n = n
        self.k = k
        self.embed_model = embed_model
        self.value_model = value_model
        self.adj_mat = adj_mat
        self.edge_mat = edge_mat
        
        self.embed_x = torch.zeros(2*n, 1, dtype=torch.float).detach()
        self.embed_u = torch.ones(2*n, conf.gnn.p_dim, dtype=torch.float).detach()
        self.embed_u = self.embed_u * 0.01
        
    def embed(self):
        for _ in range(conf.gnn.T):
            embed_u = self.embed_model(self.embed_x, 
                                       torch.mm(self.adj_mat, self.embed_u), 
                                       self.edge_mat)
    
        embed_u = (embed_u-torch.mean(embed_u))/torch.std(embed_u)
        sum_nodes = torch.mm(torch.ones(2*self.n,1), embed_u.sum(dim = 0).view(1,-1))
        return embed_u, sum_nodes
            
    
    def get_values(self, features, k): 
    
        embed_u, sum_nodes = self.embed()
        k_tensor = torch.ones(2*self.n, 1) * k
        value_tensor = self.value_model([embed_u, sum_nodes, k_tensor, 
                                         torch.cat([features,features],dim = 0)])

        return embed_u, value_tensor, sum_nodes
    
    def update_embedX(self, solution, left):
        for i in solution: self.embed_x[i] = 1
        self.embed_x[self.n:2*self.n] = 1
        for i in left: self.embed_x[i+self.n] = 0
    
    
class Log:
    def __init__(self, code):
        if not os.path.isdir(conf.path.log): os.mkdir(conf.path.log) 
        # c_d_mode_feature_gnn_concat_p-dim_best_play.txt
        self.log_name = code + '-{0}-{1}-{2}-{3}-{4}-{5}-{6}-{7}-{8}.txt'.format(
             conf.general.c, conf.general.d, conf.general.mode, 
             conf.general.feature, conf.general.gnn, conf.general.concat,
             conf.gnn.p_dim, conf.memory.best_memory_play, 
             str(datetime.datetime.now()).split('.')[0])
        self.log_pat = self.log_name.split('.')[0]
        files = open(conf.path.log + self.log_name, 'w')
        files.close()
        
    def insert(self, text):
        file_content = open(conf.path.log + self.log_name, 'r').read()
        files = open(conf.path.log + self.log_name, 'w')
        files.write(file_content)
        files.write(text + '\n')
        files.close()
        
def get_graphs(loc):
    graphs = os.listdir(conf.path.data + loc)
    return_list = []
    for graph in graphs: 
        if graph[:9] == '{0:03}_{1:03}_{2}'.format(
           conf.general.c, conf.general.d, conf.general.mode):
            return_list.append(graph)
    return return_list

class Epsilon:
    def __init__(self):
        self.cur_epi   = conf.general.cur_epi
        self.decay_epi = conf.general.decay_epi
        self.min_epi   = conf.general.min_epi
        
    def update(self):
        if self.cur_epi * self.decay_epi < self.min_epi: pass
        else: self.cur_epi *= self.decay_epi
        
    def get(self):
        return self.cur_epi 
        
        