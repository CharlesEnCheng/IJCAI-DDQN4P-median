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

import pandas as pd
import random
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from xpress_ver0417 import get_format, get_solution_pmedian
from torch.autograd import Variable
from config import conf
from utils import Log, get_graphs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# ====================================================== # 
#                                                        #
#                                                        #
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

def normalise(data):
    for col in range(data.shape[1]):
        if data[:,col].std() == 0: pass
        else: 
            data[:,col] = (data[:,col] - data[:,col].mean())/data[:,col].std()
    return data
    
def summary(aList):
    feasible_number = 0
    feasible_opt_solu = 0    
    feasible_app_solu = 0  
    infeasible_number = 0
    infeasible_app_solu= 0
    feasible_k = aList[0][3]
    
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
#                                                        #
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
        #self.fea_in = nn.Linear(10, 2*p_dim) 
        self.node_in = nn.Linear(p_dim, 3*p_dim) 
        self.sumn_in = nn.Linear(p_dim, 2*p_dim) 
        self.k_in = nn.Linear(1, 2*p_dim) 
        self.output = nn.Linear(7*p_dim, 1) 
        #self.output = nn.Linear(4*p_dim, 1) 

    def forward(self, features):
        #fea = features[0] 
        node = features[0] 
        sum_node = features[1] 
        k = features[2] 
        #k = features[1]
        #fea_layer = torch.tanh(self.fea_in(fea))
        node_layer = torch.tanh(self.node_in(node))
        sum_layer  = torch.tanh(self.sumn_in(sum_node))
        k_layer    = torch.tanh(self.k_in(k))
        layer = torch.cat((node_layer, sum_layer, k_layer), dim=1)
        #layer = torch.cat((fea_layer, k_layer), dim=1)

        output = self.output(torch.relu(layer))
        return output

def get_dist_info(instance: list, weights_model: get_embed_weights):
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

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  

class Memory:
    def __init__(self, num_features, model):
        self.num_data = 0
        self.num_fatal = 0
        
        self.max_num_data = 0
        self.max_num_fatal = 0
        
        self.loss = nn.MSELoss()
        self.batch = conf.memory.batch
        self.collect_steps = conf.memory.collect_steps
        self.show = True
        self.memory_size = conf.memory.max_cap
        
        #self.data_x_f = torch.rand(self.memory_size, num_features)
        self.data_x_u = torch.rand(self.memory_size, num_features)
        self.data_x_T = torch.rand(self.memory_size, num_features)
        self.data_x_k = torch.rand(self.memory_size, 1)
        self.data_y = torch.rand(self.memory_size, 1)
        
        self.fatal_x_u = torch.rand(self.memory_size, num_features)
        self.fatal_x_T = torch.rand(self.memory_size, num_features)
        self.fatal_x_k = torch.rand(self.memory_size, 1)
        self.fatal_y = torch.rand(self.memory_size, 1)
        
        self.model = model
        self.optimiser = optim.Adam(self.model.parameters(), lr = 0.0001)

        #self.x_f = torch.rand(self.batch, num_features)
        self.x_u = torch.rand(self.batch, num_features)
        self.x_T = torch.rand(self.batch, num_features)
        self.x_k = torch.rand(self.batch, 1)
        self.y = torch.rand(self.batch, 1)
                    
    def memorise(self, memory, target):
        #self.data_x_f[self.num_data] = memory[0].view(1,-1)
        self.data_x_u[self.num_data] = memory[0].view(1,-1)
        self.data_x_T[self.num_data] = memory[1].view(1,-1)
        self.data_x_k[self.num_data] = memory[2].view(1,-1)
        #self.data_x_k[self.num_data] = memory[1].view(1,-1)
        self.data_y[self.num_data] = target.view(1,-1)
        self.num_data += 1
        self.max_num_data += 1
        if self.num_data >= self.memory_size:
            self.num_data = 0
        
    def critical_memorise(self, memory, target):
        self.fatal_x_u[self.num_data] = memory[0].view(1,-1)
        self.fatal_x_T[self.num_data] = memory[1].view(1,-1)
        self.fatal_x_k[self.num_data] = memory[2].view(1,-1)
        self.fatal_y[self.num_fatal] = target.view(1,-1)
        self.num_fatal += 1
        self.max_num_fatal += 1
        if self.num_fatal >= self.memory_size:
            self.num_fatal = 0
        
    def replay(self):
        if self.num_data < self.collect_steps: return 
        #if self.num_data < self.batch or self.num_fatal < self.batch: return 
            
        if self.show:
            print('Model starts being trained...')
            self.show = False
            
        if self.max_num_data >= self.memory_size:
            self.max_num_data = self.memory_size
        #if self.max_num_fatal >= self.memory_size:
            #self.max_num_fatal = self.memory_size
            
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

def get_left_customer(n, instance, solution):
    served = list(set([i[1] for i in instance if i[0] in solution]))
    return list(set(range(n)).difference(set(served)))

def stop(left, solution, k):
    if left == [] and len(solution) >= k:
        return True
    else: return False

def pick_action(n, cur_epi, solution, value_tensor):
    available_actions = list(set(range(n)).difference(set(solution)))
    if np.random.rand() < cur_epi:
        return random.choice(available_actions)
    else:
        return available_actions[torch.argmax(value_tensor[available_actions])]

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
    
def get_cost(n, k, instance, left, solution):
    served = list(set(range(n)).difference(set(left)))
    cost = 0
    for node in served:
        cost -= min([i[2] for i in instance if i[1] == node and i[0] in solution])
    #if len(solution) > k: cost += (len(solution) - k)*penalty
    if len(solution) > k: cost += penalty
    return cost

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #

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
    test_graphs  = sorted(get_graphs('test/'))

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
        n = 15
        k = 5
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
            
