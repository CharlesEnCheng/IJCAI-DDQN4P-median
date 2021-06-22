#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 22:56:29 2021

@author: en-chengchang
"""

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  
from copy import deepcopy as cp
import random
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from xpress_ver0417 import get_format, get_solution_pmedian
from config import conf
from utils import Log, get_graphs, summary
from utils import get_features, normalise, update_features
from utils import agment, get_adj_matrix, get_dist_info, embed
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
    def __init__(self, p_dim):
        super().__init__()
        torch.manual_seed(conf.general.seed)      
        self.w_in = nn.Linear(1, p_dim) 
        #init.uniform(self.w_in.weight.data, -0.05, 0.05)

    def forward(self, w):
        return self.w_in(w)

class get_embedding(nn.Module):
    def __init__(self, x_in, p_dim):
        super().__init__()
        torch.manual_seed(conf.general.seed)
        self.T = conf.gnn.T
        
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
    def __init__(self, p_dim, feature_num):
        super().__init__()
        torch.manual_seed(conf.general.seed)

        self.node_in = nn.Linear(p_dim, 1*p_dim) 
        self.sumn_in = nn.Linear(p_dim, 1*p_dim) 
        self.k_in = nn.Linear(1, 1*p_dim) 
        self.fea_in = nn.Linear(feature_num, 1*p_dim) 
        self.output = nn.Linear(4*p_dim, 2*p_dim) 
        self.output2 = nn.Linear(2*p_dim, 1) 


    def forward(self, features):
        node = features[0] 
        sum_node = features[1] 
        k = features[2] 
        fea = features[3]

        node_layer = torch.tanh(self.node_in(node))
        sum_layer  = torch.tanh(self.sumn_in(sum_node))
        k_layer    = torch.tanh(self.k_in(k))
        fea_layer    = torch.tanh(self.fea_in(fea))
        layer = torch.cat((node_layer, sum_layer, k_layer, fea_layer), dim=1)

        output = self.output(torch.tanh(layer))
        output = self.output2(output)
        return output

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  
 
class Memory:
    def __init__(self, model, tar_model):
        num_features = conf.gnn.p_dim
        nf = conf.general.num_features
        
        self.num_data = 0
        self.max_num_data = 0

        self.loss = nn.MSELoss()
        self.batch = conf.memory.batch
        self.collect_steps = conf.memory.collect_steps
        self.show = True
        self.memory_size = conf.memory.max_cap
        
        self.data_x_f = torch.rand(self.memory_size, nf)
        self.data_x_u = torch.rand(self.memory_size, num_features)
        self.data_x_T = torch.rand(self.memory_size, num_features)
        self.data_x_k = torch.rand(self.memory_size, 1)
        
        self.data_y_f = [True for i in range(self.memory_size)]
        self.data_y_u = [True for i in range(self.memory_size)]
        self.data_y_T = [True for i in range(self.memory_size)]
        self.data_y_k = [True for i in range(self.memory_size)]
        
        self.data_done = torch.rand(self.memory_size, 1)
        self.data_reward = torch.rand(self.memory_size, 1)
        self.data_solution = [True for i in range(self.memory_size)]
        
        self.tar_model = tar_model
        self.model = model
        self.optimiser = optim.Adam(self.model.parameters(), lr = 0.0001)

        self.x_f = torch.rand(self.batch, nf)
        self.x_u = torch.rand(self.batch, num_features)
        self.x_T = torch.rand(self.batch, num_features)
        self.x_k = torch.rand(self.batch, 1)

        self.y = torch.rand(self.batch, 1)

        
    def memorise(self, memory, target, done, reward, solution):

        self.data_x_u[self.num_data] = memory[0].view(1,-1)
        self.data_x_T[self.num_data] = memory[1].view(1,-1)
        self.data_x_k[self.num_data] = memory[2].view(1,-1)
        self.data_x_f[self.num_data] = memory[3].view(1,-1)

        self.data_y_u[self.num_data] = target[0]#.view(1,-1)
        self.data_y_T[self.num_data] = target[1]#.view(1,-1)
        self.data_y_k[self.num_data] = target[2]#.view(1,-1)
        self.data_y_f[self.num_data] = target[3]#.view(1,-1)
        
        self.data_done[self.num_data] = done
        self.data_reward[self.num_data] = reward
        self.data_solution[self.num_data] = solution
        
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

        self.x_f[:self.batch,:] = self.data_x_f[sample_index1,:].detach()
        self.x_u[:self.batch,:] = self.data_x_u[sample_index1,:].detach()
        self.x_T[:self.batch,:] = self.data_x_T[sample_index1,:].detach()
        self.x_k[:self.batch,:] = self.data_x_k[sample_index1,:].detach()   
        
        tmp_done = self.data_done[sample_index1,:]
        tmp_reward = self.data_reward[sample_index1,:]
         
        for i in range(self.batch):
            if tmp_done[i] == True:
                self.y[:self.batch,:] = tmp_reward[i]     
            elif tmp_done[i] == False:
                tmp1 = self.data_y_u[sample_index1[i]].detach()
                tmp2 = self.data_y_T[sample_index1[i]].detach()
                tmp3 = self.data_y_k[sample_index1[i]].detach()
                tmp4 = self.data_y_f[sample_index1[i]].detach() 
                
                tmp3 = torch.ones(tmp1.shape[0], 1) * tmp3
                tmp4 = torch.cat([tmp4,tmp4],dim = 0)
                
                tmp0 = self.tar_model([tmp1, tmp2, tmp3, tmp4])
                action_ = pick_action(int(tmp1.shape[0]/2), 0, 
                                     self.data_solution[sample_index1[i]], tmp0)
                val_ = tmp0[action_].detach().clone()
                
                self.y[i] = (tmp_reward[i] + gamma*val_).detach().clone()
                del tmp1, tmp2, tmp3, tmp4, tmp0, val_
            else:
                print('There are bugs')

        out  = self.model([self.x_u.detach(), self.x_T.detach(), self.x_k.detach(), self.x_f.detach()])

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

def get_values(embed_x, embed_u, features, embed_model, value_model, 
               adj_mat, edge_mat, k): 

    embed_u, sum_nodes = embed(embed_x, embed_u, embed_model, adj_mat, edge_mat)
    k_tensor = torch.ones(len(embed_x), 1) * k
    value_tensor = value_model([embed_u, sum_nodes, k_tensor, torch.cat([features,features],dim = 0)])
    
    return embed_u, value_tensor, sum_nodes

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #

if True: # parameters and init
    cur_epi = conf.general.cur_epi
    min_epi = conf.general.min_epi
    decay_epi = conf.general.decay_epi
    
    penalty = conf.general.penalty
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
    value_model_action = ValueModel(p_dim, conf.general.num_features)
    value_model_target = ValueModel(p_dim, conf.general.num_features)
    memory = Memory(value_model_action, value_model_target)
    embedding_model = get_embedding(1, p_dim)
    weights_model = get_embed_weights(p_dim)
    log = Log('concate')


# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #

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
        
        # these are features of the 10 features
        features = get_features(n, k, instance)
        features = normalise(features)
        features = torch.tensor(features).type(torch.float).detach()  
        
        # fc_features is the embedding values of nodes
        features_f = torch.ones(n, p_dim) *0.01 
        features_c = torch.ones(n, p_dim) *0.01      
        
        fc_features = torch.cat([features_f, features_c], dim = 0)
        fc_features = fc_features.detach()
        
        last_cost = 0
        solution = []
        left = get_left_customer(n, instance, solution)
        x = torch.zeros(2*n, 1, dtype=torch.float).detach()
        discount = 0
        
        while not stop(left, solution, k):   

            features_cp = features.detach().clone()
            embed_u, value_tensor, sum_nodes = get_values(x, fc_features, features, embedding_model, 
                                               memory.model, adj_mat, edge_mat, k-discount)
            
            
            action = pick_action(n, cur_epi, solution, value_tensor)
            
            Qsa = value_tensor[action][0]
            solution.append(action)
            left = get_left_customer(n, instance, solution)
            
            for i in solution: x[i] = 1
            x[n:2*n] = 1
            for i in left: x[i+n] = 0
            
            if stop(left, solution, k):
                done = True
                reward = torch.tensor(get_cost(n, k, instance, left, solution))
            else:
                done = False
                features   = update_features(n, instance, left, features).detach()
                features = normalise(features)
             
                embed_features_, value_tensor_, sum_features_ = get_values(x, fc_features, features, embedding_model, 
                               value_model_target, adj_mat, edge_mat, k-discount-1)
            
                Qval = value_tensor_.detach()[:n]
                current_cost = get_cost(n, k, instance, left, solution)
                reward = current_cost - last_cost
                last_cost = current_cost
        
            memory_data = [embed_u[action].detach().clone(), 
                           sum_nodes[action].detach().clone(), 
                           torch.ones(1,1).detach().clone()*(k - discount),
                           features_cp[action].detach().clone()]
            
            memory_targ = [embed_features_.detach().clone(), 
                           sum_features_.detach().clone(), 
                           torch.ones(1,1).detach().clone()*(k - discount-1),
                           features.detach().clone()]
            

            memory.memorise(memory_data, memory_targ, done, reward, cp(solution))

            if steps % 4 == 0 and steps > 0:
                memory.replay()
            if steps % 100 == 0 and steps > 0:
                value_model_target.load_state_dict(memory.model.state_dict())
            
            steps += 1
            discount += 1
            del embed_u, value_tensor, sum_nodes

    del Qval
  
    
    if figure%10 == 0 and figure > 0:
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
            
            features = get_features(n, k, instance)
            features = normalise(features)
            features = torch.tensor(features).type(torch.float).detach()  
            
            features_f = torch.ones(n, p_dim) *0.01 
            features_c = torch.ones(n, p_dim) *0.01      
            
            fc_features = torch.cat([features_f, features_c], dim = 0)
            
            solution = []
            left = get_left_customer(n, instance, solution)
            x = torch.zeros(2*n, 1, dtype=torch.float)
            discount = 0
            
            while not stop(left, solution, k):   
                embed_u, value_tensor, sum_nodes = get_values(x, fc_features, features, embedding_model, 
                                               value_model_target, adj_mat, edge_mat, k-discount)

                action = pick_action(n, 0, solution, value_tensor)
                
                Qsa = value_tensor[action][0]
                solution.append(action)
                left = get_left_customer(n, instance, solution)
                
                for i in solution: x[i] = 1
                x[n:2*n] = 1
                for i in left: x[i+n] = 0
                
                features   = update_features(n, instance, left, features).detach()
                features = normalise(features)
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
            
