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
import random
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from xpress_ver0417 import get_format, get_solution_pmedian
from config import conf
from utils import Log, get_graphs
from utils import get_features, normalise, summary, update_features
from utils import get_left_customer, stop, pick_action, get_cost

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False


# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  

"""
the model structure for estimating the values of facility nodes.
variable features is like [feature] and 
feature is like
        f1,   f2,   f3,...   f10
node1
node2
...
nodeN

As such, variable value will be like
v1, v2, ... vN
"""
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
        self.max_num_data = 0
        
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
        """
        Parameters
        ----------
        memory : torch.array R 1*10
            the picked action's node representation
        target : torch.array R n*10
            the next state's node representations
        done : bool
            1 represents for the final state; 0 vice versa
        reward : torch[int]
            reward from taking(picking) an action(node)
        solution : List
            current facilities that have been picked.

        Returns
        -------
        None.

        """
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

            
        # sample from memory
        sample_index1 = random.sample(range(self.max_num_data), self.batch)
        self.x_f[:self.batch,:] = self.data_x_f[sample_index1,:].detach()
        tmp_done = self.data_done[sample_index1,:]
        tmp_reward = self.data_reward[sample_index1,:]
        
        
        for i in range(self.batch):
            # if the state is the final state, the label will be the reward
            if tmp_done[i] == True:
                self.y[:self.batch,:] = tmp_reward[i]    
            # if not, the label = reward + gamma*max(Q)
            # variable val_ = max(Q)
            elif tmp_done[i] == False:

                tmp4 = self.data_y_f[sample_index1[i]].detach() 
                tmp0 = self.tar_model([tmp4])
                
                action_ = pick_action(tmp0.shape[0], 0, 
                                      self.data_solution[sample_index1[i]], tmp0)
                val_ = tmp0[action_].detach().clone()
                
                self.y[i] = (tmp_reward[i] + gamma*val_).detach().clone()
                del tmp4, tmp0, val_
            else:
                print('There are bugs =P')

        out  = self.model([self.x_f.detach()])

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
    p_dim = conf.general.num_features
    run_valid = 10 # run validation every 10 training graphs
    
    # for recording feasible rate and the cost.
    # max_fn and max_app are the higher the better
    # when reaching higher values, the model will be saved.
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
          
        # init when every time replay
        features = get_features(n, k, instance)
        features = normalise(features)
        features = torch.tensor(features).type(torch.float).detach()  

        last_cost = 0
        solution = []
        left = get_left_customer(n, instance, solution)
        
        while not stop(left, solution, k):   
            features_cp = features.detach().clone()
            value_tensor = value_model_action([features])

            action = pick_action(n, cur_epi, solution, value_tensor)
            solution.append(action)
            left = get_left_customer(n, instance, solution)
            
            
            if stop(left, solution, k):
                done = True
                reward = torch.tensor(get_cost(n, k, instance, left, solution))
            else:
                done = False
                features = update_features(n, instance, left, features).detach()
                features = normalise(features)

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
            del value_tensor
            try:
                del embed_features_, value_tensor_, sum_features_
            except: pass

    if figure%run_valid == 0 and figure > 0:
        n = int(valid_graphs[0].split('_')[4])
        k = int(valid_graphs[0].split('_')[5].split('.')[0])
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

            while not stop(left, solution, k):   
                value_tensor = value_model_target([features])
    
                action = pick_action(n, 0, solution, value_tensor)
                solution.append(action)
                left = get_left_customer(n, instance, solution)
                
                features = update_features(n, instance, left, features).detach()
                features = normalise(features)

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
            
