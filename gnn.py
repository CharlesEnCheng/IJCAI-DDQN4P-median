#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 12:31:20 2021

@author: en-chengchang
"""


# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  

import random
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from xpress_ver0417 import get_format, get_solution_pmedian
from config import conf
from utils import Log, get_graphs, summary
from utils import agment, get_adj_matrix, get_dist_info, get_values
from utils import get_left_customer, stop, pick_action, get_cost

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  

class get_embed_weights(nn.Module):
    def __init__(self, p_dim, seed = 42):
        super().__init__()
        torch.manual_seed(seed)      
        self.w_in = nn.Linear(1, p_dim) 
        #init.uniform(self.w_in.weight.data, -0.05, 0.05)

    def forward(self, w):
        return self.w_in(w)

class get_embedding(nn.Module):
    def __init__(self, x_in, p_dim, seed = 42, T = 5):
        super().__init__()
        torch.manual_seed(seed)
        self.T = T
        
        self.x_in = nn.Linear(x_in, p_dim) 
        self.u_in = nn.Linear(p_dim, p_dim) 
        self.w_in = nn.Linear(p_dim, p_dim) 
        
        self.x_out = nn.Linear(p_dim, p_dim) 
        self.u_out = nn.Linear(p_dim, p_dim) 
        self.w_out = nn.Linear(p_dim, p_dim) 

    def forward(self, x, u, w):
  
        x_embed = self.x_out(self.x_in(x))
        u_embed = self.u_out(self.u_in(u))
        w_embed = self.w_out(self.w_in(w))
        values = x_embed + u_embed + w_embed
        return values

class ValueModel(nn.Module):
    def __init__(self, p_dim, seed = 42):
        super().__init__()
        torch.manual_seed(seed)

        self.node_in = nn.Linear(p_dim, 3*p_dim) 
        self.sumn_in = nn.Linear(p_dim, 2*p_dim) 
        self.k_in = nn.Linear(1, 2*p_dim) 
        self.output = nn.Linear(7*p_dim, 1) 


    def forward(self, features):

        node = features[0] 
        sum_node = features[1] 
        k = features[2] 

        node_layer = torch.tanh(self.node_in(node))
        sum_layer  = torch.tanh(self.sumn_in(sum_node))
        k_layer    = torch.tanh(self.k_in(k))
        layer = torch.cat((node_layer, sum_layer, k_layer), dim=1)

        output = self.output(torch.relu(layer))
        return output

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  

class Memory:
    def __init__(self, num_features, model):
        self.num_data = 0
        self.max_num_data = 0

        
        self.loss = nn.MSELoss()
        self.batch = conf.memory.batch
        self.collect_steps = conf.memory.collect_steps
        self.show = True
        self.memory_size = conf.memory.max_cap
        
        self.data_x_u = torch.rand(self.memory_size, num_features)
        self.data_x_T = torch.rand(self.memory_size, num_features)
        self.data_x_k = torch.rand(self.memory_size, 1)
        self.data_y = torch.rand(self.memory_size, 1)
           
        self.model = model
        self.optimiser = optim.Adam(self.model.parameters(), lr = 0.0001)

        self.x_u = torch.rand(self.batch, num_features)
        self.x_T = torch.rand(self.batch, num_features)
        self.x_k = torch.rand(self.batch, 1)
        self.y = torch.rand(self.batch, 1)
                    
    def memorise(self, memory, target):
        self.data_x_u[self.num_data] = memory[0].view(1,-1)
        self.data_x_T[self.num_data] = memory[1].view(1,-1)
        self.data_x_k[self.num_data] = memory[2].view(1,-1)
        self.data_y[self.num_data] = target.view(1,-1)
        self.num_data += 1
        self.max_num_data += 1
        if self.num_data >= self.memory_size:
            self.num_data = 0
        
    def replay(self):
        if self.num_data < self.collect_steps: return 
            
        if self.show:
            print('Model starts being trained...')
            self.show = False
            
        if self.max_num_data >= self.memory_size:
            self.max_num_data = self.memory_size

            
        sample_index1 = random.sample(range(self.max_num_data), self.batch)

        self.x_u[:self.batch,:] = self.data_x_u[sample_index1,:].detach()
        self.x_T[:self.batch,:] = self.data_x_T[sample_index1,:].detach()
        self.x_k[:self.batch,:] = self.data_x_k[sample_index1,:].detach()   
        
        self.y[:self.batch,:] = self.data_y[sample_index1,:].detach()
        out  = self.model([self.x_u.detach(), self.x_T.detach(), self.x_k.detach()])
        mse = self.loss(out, self.y.detach())
        self.optimiser.zero_grad()
        mse.backward(retain_graph=True)         
        self.optimiser.step()  
    
        del out, mse
        del sample_index1


# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== # 

if True: # parameters and init
    cur_epi = conf.general.cur_epi
    min_epi = conf.general.min_epi
    decay_epi = conf.general.decay_epi
    
    gamma = conf.rl.gamma
    
    train_graphs = sorted(get_graphs('train/'))
    valid_graphs = sorted(get_graphs('valid/'))

    n_replay = conf.general.n_replay
    recorded_optimal = [False for i in range(len(valid_graphs))]
    p_dim = conf.gnn.p_dim
    run_valid = 10
    
    max_fn = 0
    max_app = -float('inf')

    steps = 0
    value_model_action = ValueModel(p_dim)
    value_model_target = ValueModel(p_dim)
    memory = Memory(p_dim, value_model_action)
    embedding_model = get_embedding(1, p_dim)
    weights_model = get_embed_weights(p_dim)
    log = Log('gnn')


for figure in range(len(train_graphs)):

    n = int(train_graphs[figure].split('_')[4])
    k = int(train_graphs[figure].split('_')[5].split('.')[0])
    
    instance = np.load(conf.path.data + 'train/' + train_graphs[figure])
    instance_tmp = [[n,k]] + agment(instance, n)
    adj_mat  = get_adj_matrix(instance_tmp).detach()
    edge_mat = get_dist_info(instance_tmp, weights_model).detach()
    
    if cur_epi * decay_epi < min_epi: pass
    else: cur_epi *= decay_epi
    
    for replay in range(n_replay):
        
        fc_features = torch.ones(2*n, p_dim).detach() *0.01   
        last_cost = 0
        solution = []
        left = get_left_customer(n, instance, solution)
        x = torch.zeros(2*n, 1, dtype=torch.float).detach()
        discount = 0
        
        while not stop(left, solution, k):   

            embed_u, value_tensor, sum_nodes = get_values(x, fc_features, embedding_model, 
                                               memory.model, adj_mat, edge_mat, k-discount)
            action = pick_action(n, cur_epi, solution, value_tensor)
            solution.append(action)
            left = get_left_customer(n, instance, solution)
            
            for i in solution: x[i] = 1
            x[n:2*n] = 1
            for i in left: x[i+n] = 0
            
            if stop(left, solution, k):
                target = torch.tensor(get_cost(n, k, instance, left, solution))
            else:
              
                embed_features_, value_tensor_, sum_features_ = get_values(x, fc_features, embedding_model, 
                               value_model_target, adj_mat, edge_mat, k-discount-1)
                Qval = value_tensor_[pick_action(n, 0, solution, value_tensor_)][0]
                
                current_cost = get_cost(n, k, instance, left, solution)
                reward = current_cost - last_cost
                target = reward + gamma * Qval
                last_cost = current_cost
            target = target.detach()
        
            
            memory_data = [embed_u[action].detach().clone(), 
                           sum_nodes[action].detach().clone(), 
                           torch.ones(1,1).detach().clone()*(k - discount)]
            
            memory.memorise(memory_data, target)
            
            if steps % 4 == 0 and steps > 0:
                memory.replay()
            if steps % 100 == 0 and steps > 0:
                value_model_target.load_state_dict(memory.model.state_dict())
            
            steps += 1
            discount += 1
            del embed_u, value_tensor, sum_nodes
            try:
                del embed_features_, value_tensor_, sum_features_
            except: pass

    del Qval
   
    
    if figure%run_valid == 0 and figure > 0:
        n = int(valid_graphs[0].split('_')[4])
        k = int(valid_graphs[0].split('_')[5].split('.')[0])
        valid_info = []
        
        for valid_data in range(len(valid_graphs)):
            instance = np.load(conf.path.data + 'valid/' + valid_graphs[valid_data])
            instance_tmp = [[n,k]] + agment(instance, n)
            adj_mat  = get_adj_matrix(instance_tmp).detach()
            edge_mat = get_dist_info(instance_tmp, weights_model).detach()
    
            if figure == run_valid:
                c = get_format(np.array(instance), n)
                obj, _y, _x = get_solution_pmedian(c, n, k)
                recorded_optimal[valid_data] = -obj
                
            optimal =  recorded_optimal[valid_data]  
            fc_features = torch.ones(2*n, p_dim).detach() *0.01   
            
            solution = []
            left = get_left_customer(n, instance, solution)
            x = torch.zeros(2*n, 1, dtype=torch.float)
            discount = 0
            
            while not stop(left, solution, k):   
                embed_u, value_tensor, sum_nodes = get_values(x, fc_features, embedding_model, 
                                               value_model_target, adj_mat, edge_mat, k-discount)
                
                action = pick_action(n, 0, solution, value_tensor)
                solution.append(action)
                left = get_left_customer(n, instance, solution)
                
                for i in solution: x[i] = 1
                x[n:2*n] = 1
                for i in left: x[i+n] = 0
                discount += 1
                
            cost = get_cost(n, k, instance, left, solution)
            valid_info.append([valid_data, optimal, cost, k, len(solution)])
            
        text, fn, avgapp = summary(valid_info)
        print('Figure:{0:04} '.format(figure)+text)
        log.insert('Figure:{0:04} '.format(figure)+text)
        
        if memory.show == False:
            if fn > max_fn:
                max_fn = fn
                max_app = avgapp
                torch.save(value_model_target.state_dict(), conf.path.model + log.log_pat)
            elif fn == max_fn and avgapp > max_app:
                max_app = avgapp
                torch.save(value_model_target.state_dict(), conf.path.model + log.log_pat)
            
