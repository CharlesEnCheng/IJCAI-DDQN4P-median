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
import gc
import sys
import pandas as pd
from tqdm import tqdm
from copy import deepcopy as cp
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import init
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

def get_instance(n, k, seed = None):

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
          
        c_nodes = random.sample(range(n), np.random.randint(int(n/k), int(2*n/k)))
        for c_node in c_nodes:
            instance.append([s, c_node, np.random.randint(20,50)])
        """
        c_nodes = random.sample(range(n), np.random.randint(1, n))
        for c_node in c_nodes:
            instance.append([s, c_node, np.random.randint(15,30)])
        """   
    return sorted(instance)

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
        #self.fea_in = nn.Linear(10, 2*p_dim) 
        self.node_in = nn.Linear(p_dim, 1*p_dim) 
        self.sumn_in = nn.Linear(p_dim, 1*p_dim) 
        self.k_in = nn.Linear(1, 1*p_dim) 
        self.fea_in = nn.Linear(feature_num, 1*p_dim) 
        self.output = nn.Linear(4*p_dim, 2*p_dim) 
        self.output2 = nn.Linear(2*p_dim, 1) 
        #self.output = nn.Linear(4*p_dim, 1) 

    def forward(self, features):
        #fea = features[0] 
        node = features[0] 
        sum_node = features[1] 
        k = features[2] 
        fea = features[3]
        #k = features[1]
        #fea_layer = torch.tanh(self.fea_in(fea))
        node_layer = torch.tanh(self.node_in(node))
        sum_layer  = torch.tanh(self.sumn_in(sum_node))
        k_layer    = torch.tanh(self.k_in(k))
        fea_layer    = torch.tanh(self.fea_in(fea))
        layer = torch.cat((node_layer, sum_layer, k_layer, fea_layer), dim=1)
        #layer = torch.cat((fea_layer, k_layer), dim=1)

        output = self.output(torch.tanh(layer))
        output = self.output2(output)
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
    
    # p_dim 先手刻為 10
    edge_mat = torch.zeros(n, p_dim)
    for row in range(n): edge_mat[row] = node_edge_val[row]
    return edge_mat

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
        self.num_fatal = 0
        
        self.max_num_data = 0
        self.max_num_fatal = 0
        
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
        
        self.fatal_x_u = torch.rand(self.memory_size, num_features)
        self.fatal_x_T = torch.rand(self.memory_size, num_features)
        self.fatal_x_k = torch.rand(self.memory_size, 1)
        self.fatal_y = torch.rand(self.memory_size, 1)
        
        self.tar_model = tar_model
        self.model = model
        self.optimiser = optim.Adam(self.model.parameters(), lr = 0.0001)

        self.x_f = torch.rand(self.batch, nf)
        self.x_u = torch.rand(self.batch, num_features)
        self.x_T = torch.rand(self.batch, num_features)
        self.x_k = torch.rand(self.batch, 1)

        #self.y_f = torch.rand(self.batch, nf)
        #self.y_u = torch.rand(self.batch, num_features)
        #self.y_T = torch.rand(self.batch, num_features)
        #self.y_k = torch.rand(self.batch, 1)
        
        self.y = torch.rand(self.batch, 1)
        
        #[True for i in range(self.batch)]
        
    def memorise(self, memory, target, done, reward, solution):
        #self.data_x_f[self.num_data] = memory[0].view(1,-1)
        self.data_x_u[self.num_data] = memory[0].view(1,-1)
        self.data_x_T[self.num_data] = memory[1].view(1,-1)
        self.data_x_k[self.num_data] = memory[2].view(1,-1)
        self.data_x_f[self.num_data] = memory[3].view(1,-1)
        #self.data_x_k[self.num_data] = memory[1].view(1,-1)
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
        #sample_index2 = random.sample(range(self.max_num_fatal), self.batch)
        
        #memory_x = self.data_x[sample_index1,:]
        #memory_y = self.data_y[sample_index1,:]
        #critical_x = self.fatal_x[sample_index2,:]
        #critical_y = self.fatal_y[sample_index2,:]

        #x = torch.cat((memory_x, critical_x), dim = 0).type(torch.float).detach().clone()       
        #y = torch.cat((memory_y, critical_y), dim = 0).type(torch.float).detach().clone()       

        self.x_f[:self.batch,:] = self.data_x_f[sample_index1,:].detach()
        self.x_u[:self.batch,:] = self.data_x_u[sample_index1,:].detach()
        self.x_T[:self.batch,:] = self.data_x_T[sample_index1,:].detach()
        self.x_k[:self.batch,:] = self.data_x_k[sample_index1,:].detach()   
        
        tmp_done = self.data_done[sample_index1,:]
        tmp_reward = self.data_reward[sample_index1,:]
        
        #self.x_u[self.batch:,:] = self.fatal_x_u[sample_index2,:].detach()
        #self.x_T[self.batch:,:] = self.fatal_x_T[sample_index2,:].detach()
        #self.x_k[self.batch:,:] = self.fatal_x_k[sample_index2,:].detach()
        
        """
        tmp1 = self.data_y_u[sample_index1,:].detach()
        tmp2 = self.data_y_T[sample_index1,:].detach()
        tmp3 = self.data_y_k[sample_index1,:].detach()
        tmp4 = self.data_y_f[sample_index1,:].detach() 
        
        self.y[:self.batch,:] = self.tar_model([tmp1, tmp2, tmp3, tmp4])
        """
        
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
                                     self.data_solution[sample_index1[i]], 
                                     tmp0)
                val_ = tmp0[action_].detach().clone()
                
                self.y[i] = (tmp_reward[i] + gamma*val_).detach().clone()
                del tmp1, tmp2, tmp3, tmp4, tmp0, val_
            else:
                print('There are bugs')
            

        #self.y[:self.batch,:] = self.data_y[sample_index1,:].detach()
        #self.y[self.batch:,:] = self.fatal_y[sample_index2,:].detach()

        #self.x_u.requires_grad = True
        #self.x_T.requires_grad = True
        #self.x_k.requires_grad = True
        
        out  = self.model([self.x_u.detach(), self.x_T.detach(), self.x_k.detach(), self.x_f.detach()])
        #out  = self.model([self.x_f.detach(), self.x_u.detach(), self.x_T.detach(), self.x_k.detach()])
        #out  = self.model([self.x_f.detach(), self.x_k.detach()])
        #out  = self.model([self.x_f.detach()])
        
        mse = self.loss(out, self.y.detach())
        self.optimiser.zero_grad()
        mse.backward(retain_graph=True)         
        self.optimiser.step()  
    
        """
        import gc
        cnt = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    cnt += 1
            except:
                pass
        print(cnt)
        """

        del out, mse
        del sample_index1#, sample_index2
        
 
        #del memory_x, memory_y, critical_x, critical_y


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
    test_graphs  = sorted(get_graphs('test/'))

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
  
    #best_play = [[False, torch.tensor(-float('inf'))]]
    if cur_epi * decay_epi < min_epi: pass
    else: cur_epi *= decay_epi

    for replay in range(n_replay):
        
        features = get_features(n, k, instance)
        features = normalise(features)
        features = torch.tensor(features).type(torch.float).detach()  
        
        features_f = torch.ones(n, p_dim) *0.01 
        features_c = torch.ones(n, p_dim) *0.01      
        
        fc_features = torch.cat([features_f, features_c], dim = 0)
        fc_features = fc_features.detach()
        
        #play_seq_tmp = []
        last_cost = 0
        solution = []
        left = get_left_customer(n, instance, solution)
        x = torch.zeros(2*n, 1, dtype=torch.float).detach()
        discount = 0
        
        while not stop(left, solution, k):   

            features_cp = features.detach().clone()
            embed_u, value_tensor, sum_nodes = get_values(x, fc_features, features, embedding_model, 
                                               memory.model, adj_mat, edge_mat, k-discount)
            

            #embed_features = embedding_model(x, fc_features, adj_mat)
            #embed_features = embed_features[:n]
            #sum_features = torch.mm(torch.ones(n,1),embed_features.sum(dim=0).view(1,-1))
            
            #value_tensor = memory.model([embed_features, sum_features, torch.ones(n,1)*(k - discount)])
            #value_tensor = memory.model([features, torch.ones(n,1)*(k - discount)])
            #value_tensor = memory.model([features])
 
            
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
                #features[:,-2] -= 1

                #embed_features_ = embedding_model(x, fc_features, adj_mat)
                #embed_features_ = embed_features_[:n]
                #sum_features_ = torch.mm(torch.ones(n,1),embed_features_.sum(dim=0).view(1,-1))
                embed_features_, value_tensor_, sum_features_ = get_values(x, fc_features, features, embedding_model, 
                               value_model_target, adj_mat, edge_mat, k-discount-1)
            
                Qval = value_tensor_.detach()[:n]
                #Qval = value_model_target([embed_features_[:n], sum_features_[:n], torch.ones(n,1)*(k - discount-1)])
                #Qval = value_model_target([features, torch.ones(n,1)*(k - discount-1)])
                #Qval = value_model_target([features])

                #Qval = Qval[pick_action(n, 0, solution, Qval)][0]
                current_cost = get_cost(n, k, instance, left, solution)
                reward = current_cost - last_cost
                #target = reward + gamma * Qval
                #target = -1 + gamma * Qval
                last_cost = current_cost
            #target = target.detach()
        
            
            memory_data = [embed_u[action].detach().clone(), 
                           sum_nodes[action].detach().clone(), 
                           torch.ones(1,1).detach().clone()*(k - discount),
                           features_cp[action].detach().clone()]
            
            memory_targ = [embed_features_.detach().clone(), 
                           sum_features_.detach().clone(), 
                           torch.ones(1,1).detach().clone()*(k - discount-1),
                           features.detach().clone()]
            
            """
            memory_data = [features[action],
                           torch.ones(1,1)*(k - discount)]
            """
            memory.memorise(memory_data, memory_targ, done, reward, cp(solution))
            #memory_data = [features[action]]
            """
            for _ in range(10000):
                memory.memorise(memory_data, memory_targ, done, reward, solution)
            
            """
            #play_seq_tmp.append([memory_data, target.detach()])
            
            if steps % 4 == 0 and steps > 0:
                memory.replay()
            if steps % 100 == 0 and steps > 0:
                value_model_target.load_state_dict(memory.model.state_dict())
            
            steps += 1
            discount += 1
            del embed_u, value_tensor, sum_nodes
            """
            try:
                del embed_features_, value_tensor_, sum_features_
            except: pass
            """
        """    
        if play_seq_tmp[-1][-1] > best_play[-1][-1]:
            best_play = play_seq_tmp
        

    for data in best_play:
        memory.critical_memorise(data[0], data[1])
    """
    del Qval
    #del features, features_cp, value_tensor, Qval#, best_play
    #gc.collect()
        
    
    if figure%10 == 0 and figure > 0:
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
                #features[:,-2] -= 1
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
            
