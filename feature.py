#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 20:47:45 2021

@author: en-chengchang
"""

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  

from copy import deepcopy as cp
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

class ValueModel(nn.Module):
    def __init__(self, feature_size, seed = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.layer1 = nn.Linear(feature_size, feature_size * 2) 
        self.layer2 = nn.Linear(feature_size * 2, feature_size * 6) 
        self.layer3 = nn.Linear(feature_size * 6, feature_size * 3) 
        self.output = nn.Linear(feature_size * 3, 1) 


    def forward(self, features):
        value = torch.tanh(self.layer1(features[0]))
        value = torch.relu(self.layer2(value))
        value = torch.relu(self.layer3(value))
        value = self.output(value)
        return value


# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  

class Memory:
    def __init__(self, nf, model, tar_model):
        self.num_data = 0
        self.num_fatal = 0
        
        self.max_num_data = 0
        self.max_num_fatal = 0
        
        self.loss = nn.MSELoss()
        self.batch = conf.memory.batch
        self.collect_steps = conf.memory.collect_steps
        self.show = True
        self.memory_size = 10**5
        
        self.data_x_f = torch.rand(self.memory_size, nf)
        self.data_y_f = [True for i in range(self.memory_size)]

        self.data_done = torch.rand(self.memory_size, 1)
        self.data_reward = torch.rand(self.memory_size, 1)
        self.data_solution = [True for i in range(self.memory_size)]
        
        
        self.tar_model = tar_model
        self.model = model
        self.optimiser = optim.Adam(self.model.parameters(), lr = 0.0001)

        self.x_f = torch.rand(self.batch, nf)
        self.y = torch.rand(self.batch, 1)
        
    def memorise(self, memory, target, done, reward, solution):
        self.data_x_f[self.num_data] = memory[0].view(1,-1)
        self.data_y_f[self.num_data] = target[0]
        
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

        tmp_done = self.data_done[sample_index1,:]
        tmp_reward = self.data_reward[sample_index1,:]
        
        
        for i in range(self.batch):
            if tmp_done[i] == True:
                self.y[:self.batch,:] = tmp_reward[i]     
            elif tmp_done[i] == False:

                tmp4 = self.data_y_f[sample_index1[i]].detach() 
                tmp0 = self.tar_model([tmp4])
                
                action_ = pick_action(tmp0.shape[0], 0, 
                                     self.data_solution[sample_index1[i]], 
                                     tmp0)
                val_ = tmp0[action_].detach().clone()
                
                self.y[i] = (tmp_reward[i] + gamma*val_).detach().clone()
                del tmp4, tmp0, val_
            else:
                print('There are bugs')

        out  = self.model([self.x_f.detach()])

        mse = self.loss(out, self.y.detach())
        self.optimiser.zero_grad()
        mse.backward(retain_graph=True)         
        self.optimiser.step()  

        del out, mse
        del sample_index1#, sample_index2
        
 

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
    p_dim = conf.general.num_features
    run_valid = 10
    
    max_fn = 0
    max_app = -float('inf')

    steps = 0
    value_model_action = ValueModel(p_dim)
    value_model_target = ValueModel(p_dim)
    memory = Memory(p_dim, value_model_action, value_model_target)
    log = Log('feature')


# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== # 

for figure in range(len(train_graphs)):
   
    n = int(train_graphs[figure].split('_')[4])
    k = int(train_graphs[figure].split('_')[5].split('.')[0])
    
    instance = np.load(conf.path.data + 'train/' + train_graphs[figure])

    if cur_epi * decay_epi < min_epi: pass
    else: cur_epi *= decay_epi
    
    for replay in range(n_replay):
          
        features = get_features(n, k, instance)
        features = normalise(features)
        features = torch.tensor(features).type(torch.float).detach()  

        last_cost = 0
        solution = []
        left = get_left_customer(n, instance, solution)
        x = torch.zeros(2*n, 1, dtype=torch.float).detach()
        discount = 0
        
        while not stop(left, solution, k):   
            features_cp = features.detach().clone()
            value_tensor = value_model_action([features])

            action = pick_action(n, cur_epi, solution, value_tensor)
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
                features = update_features(n, instance, left, features).detach()
                features = normalise(features)
                #features[:,-2] -= 1

                current_cost = get_cost(n, k, instance, left, solution)
                reward = current_cost - last_cost
                last_cost = current_cost

            memory_data = [features_cp[action].detach().clone()]
            memory_targ = [features.detach().clone()]
            
            memory.memorise(memory_data, memory_targ, done, reward, cp(solution))
            
            if steps % 4 == 0 and steps > 0:
                memory.replay()
            if steps % 100 == 0 and steps > 0:
                value_model_target.load_state_dict(memory.model.state_dict())
            
            steps += 1
            discount += 1
            del value_tensor
            try:
                del embed_features_, value_tensor_, sum_features_
            except: pass

    if figure%run_valid == 0 and figure > 0:
        n = 15
        k = 5
        valid_info = []
        
        for valid_data in range(len(valid_graphs)):
            instance = np.load(conf.path.data + 'valid/' + valid_graphs[valid_data])

            if figure == run_valid:
                c = get_format(np.array(instance), n)
                obj, _y, _x = get_solution_pmedian(c, n, k)
                recorded_optimal[valid_data] = -obj
                
            optimal =  recorded_optimal[valid_data]  

            features = get_features(n, k, instance)
            features = normalise(features)
            features = torch.tensor(features).type(torch.float).detach()  
    
            last_cost = 0
            solution = []
            left = get_left_customer(n, instance, solution)
            x = torch.zeros(2*n, 1, dtype=torch.float).detach()
            discount = 0
            
            while not stop(left, solution, k):   
                value_tensor = value_model_target([features])
    
                action = pick_action(n, 0, solution, value_tensor)
                solution.append(action)
                left = get_left_customer(n, instance, solution)
                
                for i in solution: x[i] = 1
                x[n:2*n] = 1
                for i in left: x[i+n] = 0
                
                features = update_features(n, instance, left, features).detach()
                features = normalise(features)
                #features[:,-2] -= 1
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
            
