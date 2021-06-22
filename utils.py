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
from config import conf

import pandas as pd
import numpy as np
import random
import torch
from torch.autograd import Variable

# ====================================================== # 
#                                                        #
#      common functions - utils                          #
#                                                        #
# ====================================================== #  
    
class Log:
    def __init__(self, code):
        if not os.path.isdir(conf.path.log): os.mkdir(conf.path.log) 
        if not os.path.isdir(conf.path.model): os.mkdir(conf.path.model) 

        self.log_name = code + '-{0}-{1}-{2}-{3}-{4}.txt'.format(
             conf.general.c, conf.general.d, conf.general.mode, 
             conf.gnn.p_dim, str(datetime.datetime.now()).split('.')[0])
        self.log_pat = self.log_name.split('.')[0]
        files = open(conf.path.log + self.log_name, 'w')
        files.close()
        
    def insert(self, text):
        file_content = open(conf.path.log + self.log_name, 'r').read()
        files = open(conf.path.log + self.log_name, 'w')
        files.write(file_content)
        files.write(text + '\n')
        files.close()
        
# fetch instance filenames matching the given c, d, mode, [train/test/valid]
def get_graphs(loc):
    graphs = os.listdir(conf.path.data + loc)
    return_list = []
    for graph in graphs: 
        if graph[:9] == '{0:03}_{1:03}_{2}'.format(
           conf.general.c, conf.general.d, conf.general.mode):
            return_list.append(graph)
    return return_list

# normalise node representations. if std = 0, skip the column
def normalise(data):
    for col in range(data.shape[1]):
        if data[:,col].std() == 0: pass
        else: 
            data[:,col] = (data[:,col] - data[:,col].mean())/data[:,col].std()
    return data
    
# show the numerical results of valid data during the training process
def summary(aList):
    feasible_number = 0
    feasible_opt_solu = 0    
    feasible_app_solu = 0  
    infeasible_number = 0
    infeasible_app_solu= 0
    
    for row in aList:
        if row[4] > row[3]: 
            infeasible_number += 1
            infeasible_app_solu += row[4]
        else: 
            feasible_number += 1
            feasible_opt_solu += row[1]
            feasible_app_solu += row[2]
       
    try:  avgOpt = str(round(feasible_opt_solu/ feasible_number, 2)) 
    except: avgOpt = 'nan'
    avgOpt = (8-len(avgOpt))*' ' + avgOpt  
    try:  avgApp = str(round(feasible_app_solu/ feasible_number, 2)) 
    except: avgApp = 'nan'
    avgApp = (8-len(avgApp))*' ' + avgApp  
    try:  avginfK = str(round(infeasible_app_solu/ infeasible_number, 1)) 
    except: avginfK = 'nan'
    avginfK = (4-len(avginfK))*' ' + avginfK  
    
    text = 'FN:{0:03}; FAvgOpt:{1}; FAvgApp:{2}; IFN:{3:03}; IFAvgK:{4};'\
        .format(feasible_number, avgOpt, avgApp, infeasible_number, avginfK)
    return text, feasible_number, float(avgApp)
    
# ====================================================== # 
#                                                        #
#      feature related functions                         #
#                                                        #
# ====================================================== # 

# create the 10 features for a graph at the initial node representations.
def get_features(n, k, instance):
    
    instance_df = pd.DataFrame(instance, columns = ['facility', 'customer', 'edge'])
    neighbour_num = instance_df.groupby(['facility'])['customer'].count().values
    
    percent_paras = [1, 25, 50, 75, 99]
    percent_array = np.zeros((n, len(percent_paras)))
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
    features = np.zeros((n, 5 + len(percent_paras)))
    features[:,0] = neighbour_num
    features[:,1:1+len(percent_paras)] = percent_array
    features[:,1+len(percent_paras)] = neighbour_num # at first, they are not satisified
    features[:,2+len(percent_paras)] = left_weights
    features[:,3+len(percent_paras)] = k    
    features[:,4+len(percent_paras)] = min_support    
        
    return features


def update_features(n, instance, left, features):
    left_tmp = [i for i in instance if i[1] in left]
    left_feature = torch.zeros(n)
    left_weights = torch.zeros(n)
    left_support_customer = torch.zeros(n)
    left_support = torch.ones(n) * n
    for i in left_tmp: 
        left_feature[i[0]] += 1
        left_weights[i[0]] += i[2]
        left_support_customer[i[1]] += 1
        
    for i in left_tmp:
        if left_support_customer[i[1]] < left_support[i[0]]:
            left_support[i[0]] = left_support_customer[i[1]]

    features[:,-4] = left_feature
    features[:,-3] = left_weights
    features[:,-1] = left_support
    return features


# ====================================================== # 
#                                                        #
#      gnn related functions                             #
#                                                        #
# ====================================================== #

def agment(instance, n):
    instance2 = np.array(instance)
    instance2[:,1] += n
    
    instance3 = []
    for i in range(n, 2*n):
        for row in instance2:
            if row[1] == i:
                instance3.append([row[1], row[0], row[2]])
    instance3 = np.array(instance3)
    instance4 = np.concatenate((instance2, instance3), axis= 0)
    instance4 = instance4.tolist()
    return instance4

def get_adj_matrix(instance: list):
    n = instance[0][0] * 2
    adj_mat = torch.zeros(n, n)
    for i, j, val in instance[1:]:
        adj_mat[i][j] = 1
    adj_mat = adj_mat
    return adj_mat

def get_dist_info(instance, weights_model):
    n = instance[0][0] * 2
    weights = np.array([i[2] for i in instance[1:]])
    weights = torch.tensor(weights.reshape(-1,1), requires_grad=False)
    weights = weights.type(torch.float)
    weights = weights_model(weights)
    
    node_edge_val = {}
    for index in range(1, len(instance)):
        if instance[index][0] not in node_edge_val.keys(): 
            node_edge_val[instance[index][0]] = []
        node_edge_val[instance[index][0]].append(weights[index-1])
    for index in range(n): 
        node_edge_val[index] = sum(node_edge_val[index])
    
    edge_mat = torch.zeros(n, conf.gnn.p_dim)
    for row in range(n): edge_mat[row] = node_edge_val[row]
    return edge_mat

def embed(embed_x, embed_u, embed_model, adj_mat, edge_mat):
    n = len(embed_x)
    T = conf.gnn.T
    for _ in range(T):
        embed_u = embed_model(embed_x, torch.mm(adj_mat, embed_u), edge_mat)
    
    embed_u = (embed_u-torch.mean(embed_u))/torch.std(embed_u)
    sum_nodes = torch.mm(torch.ones(n,1), embed_u.sum(dim = 0).view(1,-1))
    return embed_u, sum_nodes
        
def get_values(embed_x, embed_u, embed_model, value_model, 
               adj_mat, edge_mat, k): 

    embed_u, sum_nodes = embed(embed_x, embed_u, embed_model, adj_mat, edge_mat)
    k_tensor = torch.ones(len(embed_x), 1) * k
    value_tensor = value_model([embed_u, sum_nodes, k_tensor])
    
    return embed_u, value_tensor, sum_nodes


# ====================================================== # 
#                                                        #
#      rl related functions                              #
#                                                        #
# ====================================================== # 

# get customers that haven't been supplied by any picked facilities.
def get_left_customer(n, instance, solution):
    served = list(set([i[1] for i in instance if i[0] in solution]))
    return list(set(range(n)).difference(set(served)))

# stop criteria
def stop(left, solution, k):
    if left == [] and len(solution) >= k:
        return True
    else: return False

# given number of n (n facilities/customers) , current solution
# and values of facilitiy nodes, return an action          
def pick_action(n, cur_epi, solution, value_tensor):
    available_actions = list(set(range(n)).difference(set(solution)))
    if np.random.rand() < cur_epi:
        return random.choice(available_actions)
    else:
        return available_actions[torch.argmax(value_tensor[available_actions])]

    
# for those customers that have been served, pick the lowest weight from 
# their picked suppliers/facilities.
def get_cost(n, k, instance, left, solution):
    served = list(set(range(n)).difference(set(left)))
    cost = 0
    for node in served:
        cost -= min([i[2] for i in instance if i[1] == node and i[0] in solution])
    #if len(solution) > k: cost += (len(solution) - k)*penalty
    if len(solution) > k: cost += conf.general.penalty
    return cost


# ====================================================== # 
#                                                        #
#      tuned-parameter gnn functions                     #
#                                                        #
# ====================================================== # 

class MySpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sp_mat, dense_mat):
        ctx.save_for_backward(sp_mat, dense_mat)

        return torch.mm(sp_mat, dense_mat)

    @staticmethod
    def backward(ctx, grad_output):        
        sp_mat, dense_mat = ctx.saved_tensors
        grad_matrix1 = grad_matrix2 = None

        assert not ctx.needs_input_grad[0]
        if ctx.needs_input_grad[1]:
            grad_matrix2 = Variable(torch.mm(sp_mat.data.t(), grad_output.data))
        
        return grad_matrix1, grad_matrix2

def gnn_spmm(sp_mat, dense_mat):
    return MySpMM.apply(sp_mat, dense_mat)
