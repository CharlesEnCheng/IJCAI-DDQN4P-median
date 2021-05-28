#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:42:29 2021

@author: en-chengchang
"""

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  

import random
import pandas as pd
import numpy as np
import torch
from config import conf

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== # 


def pick_action(n, cur_epi, solution, value_tensor, seed):
    
    np.random.seed(seed=seed)
    random.seed(seed)
        
    available_actions = list(set(range(n)).difference(set(solution)))
    if np.random.rand() < cur_epi: return random.choice(available_actions)
    else: return available_actions[torch.argmax(value_tensor[available_actions])]


def normalise(data):
    for col in range(data.shape[1]):
        if data[:,col].std() == 0: pass
        else: 
            data[:,col] = (data[:,col] - data[:,col].mean())/data[:,col].std()
    return data


class Data:
    def __init__(self, instance, n, k):
        self.instance = instance
        self.n = n
        self.k = k
        self.solution = []
        
    def reset(self):
        self.solution = []
        self.features = self.get_features()
        
    def augment(self):
        instance2 = np.array(self.instance)
        instance2[:,1] += self.n
        
        instance3 = []
        for i in range(self.n, 2*self.n):
            for row in instance2:
                if row[1] == i:
                    instance3.append([row[1], row[0], row[2]])
        instance3 = np.array(instance3)
        instance4 = np.concatenate((instance2, instance3), axis= 0)
        instance4 = instance4.tolist()
        self.aug_instance = instance4
        return instance4
    
    def get_adj_matrix(self):
        n = self.n * 2
        adj_mat = torch.zeros(n, n)
        for i, j, val in self.aug_instance:
            adj_mat[i][j] = 1
        self.adj_mat = adj_mat
        return adj_mat
    
    def get_features(self):
        
        instance_df = pd.DataFrame(self.instance, columns = ['facility', 'customer', 'edge'])
        neighbour_num = instance_df.groupby(['facility'])['customer'].count().values
        
        percent_paras = [1, 25, 50, 75, 99]
        percent_array = np.zeros((self.n, len(percent_paras)))
        index = 0
        for percent in percent_paras:
            percent_array[:,index] = instance_df.groupby(['facility'])['edge']\
                                   .apply(list).apply(lambda x: np.percentile(x, percent))
            index += 1
        min_support_dict = instance_df.groupby(['customer'])['facility'].count().reset_index().values
        min_support_dict = {i[0]:i[1] for i in min_support_dict}
        tmp_support = instance_df.groupby(['facility'])['customer'].apply(list).apply(lambda x:[min_support_dict[i] for i in x])
        min_support = tmp_support.apply(lambda x: min(x)).values
        
        left_weights = instance_df.groupby(['facility'])['edge'].sum().values
        features = np.zeros((self.n, 5 + len(percent_paras)))
        features[:,0] = neighbour_num
        features[:,1:1+len(percent_paras)] = percent_array
        features[:,1+len(percent_paras)] = neighbour_num # at first, they are not satisified
        features[:,2+len(percent_paras)] = left_weights
        features[:,3+len(percent_paras)] = self.k    
        features[:,4+len(percent_paras)] = min_support    
        self.features = features   
        
        assert self.features.shape[1] == conf.general.num_features, "must be introducing new features but the config file hasn't been updated"
        return features
    
    def update_features(self):
        left_tmp = [i for i in self.instance if i[1] in self.left]
        left_feature = torch.zeros(self.n)
        left_weights = torch.zeros(self.n)
        left_support_customer = torch.zeros(self.n)
        left_support = torch.ones(self.n) * self.n
        for i in left_tmp: 
            left_feature[i[0]] += 1
            left_weights[i[0]] += i[2]
            left_support_customer[i[1]] += 1
            
        for i in left_tmp:
            if left_support_customer[i[1]] < left_support[i[0]]:
                left_support[i[0]] = left_support_customer[i[1]]
    
        self.features[:,-4] = left_feature
        self.features[:,-3] = left_weights
        self.features[:,-1] = left_support
        return self.features  
       
    def get_embed_weights(self, weights_model):
        n = self.n *2

        weights = np.array([i[2] for i in self.aug_instance])
        weights = torch.tensor(weights.reshape(-1,1), requires_grad=False)
        weights = weights.type(torch.float)
        weights = weights_model(weights)
        
        node_edge_val = {}
        for index in range(len(self.aug_instance)):
            if self.aug_instance[index][0] not in node_edge_val.keys(): 
                node_edge_val[self.aug_instance[index][0]] = []
            node_edge_val[self.aug_instance[index][0]].append(weights[index-1])
        for index in range(n): 
            node_edge_val[index] = sum(node_edge_val[index])
        
        edge_mat = torch.zeros(n, conf.gnn.p_dim)
        for row in range(n): edge_mat[row] = node_edge_val[row]
        self.edge_mat = edge_mat
        return edge_mat
    
    def pick_action(self, cur_epi, value_tensor):
        return pick_action(self.n, cur_epi, self.solution, 
                           value_tensor, seed = int(self.instance[0][2]*self.n/(len(self.solution)+1)))
    
    def get_left_customer(self):
        served = list(set([i[1] for i in self.instance if i[0] in self.solution]))
        self.left = list(set(range(self.n)).difference(set(served)))
        return self.left

    def stop(self):
        _ = self.get_left_customer()
        if self.left == [] and len(self.solution) >= self.k: return True
        else: return False    
        
    def get_cost(self):
        served = list(set(range(self.n)).difference(set(self.left)))
        cost = 0
        for node in served:
            cost -= min([i[2] for i in self.instance if i[1] == node and i[0] in self.solution])
        if len(self.solution) > self.k: cost += conf.general.penalty
        return cost