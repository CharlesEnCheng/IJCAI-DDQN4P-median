#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 17:21:38 2021

@author: en-chengchang
"""

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  

import random
import torch
from torch import nn
import torch.optim as optim
from config import conf
from instance_utils import pick_action

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  

class Memory:
    def __init__(self, num_features, act_model, tar_model):
        self.num_data = 0; self.num_fatal = 0
        self.max_num_data = 0; self.max_num_fatal = 0
        
        self.loss = nn.MSELoss()
        if conf.memory.best_memory_play == True:
            self.batch = int(conf.memory.batch/2)
        else:
            self.batch = conf.memory.batch
        self.collect_steps = conf.memory.collect_steps
        self.show = True
        self.memory_size = conf.memory.max_cap
        
        self.data_x_f = torch.rand(self.memory_size, num_features)
        self.data_x_u = torch.rand(self.memory_size, conf.gnn.p_dim)
        self.data_x_T = torch.rand(self.memory_size, conf.gnn.p_dim)
        self.data_x_k = torch.rand(self.memory_size, 1)
        
        self.data_y_f = [True for i in range(self.memory_size)]
        self.data_y_u = [True for i in range(self.memory_size)]
        self.data_y_T = [True for i in range(self.memory_size)]
        self.data_y_k = [True for i in range(self.memory_size)]
        
        self.data_done = torch.rand(self.memory_size, 1)
        self.data_reward = torch.rand(self.memory_size, 1)
        self.data_solution = [True for i in range(self.memory_size)]
        
        self.fatal_x_f = torch.rand(self.memory_size, num_features)
        self.fatal_x_u = torch.rand(self.memory_size, conf.gnn.p_dim)
        self.fatal_x_T = torch.rand(self.memory_size, conf.gnn.p_dim)
        self.fatal_x_k = torch.rand(self.memory_size, 1)
        
        self.fatal_y_f = [True for i in range(self.memory_size)]
        self.fatal_y_u = [True for i in range(self.memory_size)]
        self.fatal_y_T = [True for i in range(self.memory_size)]
        self.fatal_y_k = [True for i in range(self.memory_size)]
        
        self.fatal_done = torch.rand(self.memory_size, 1)
        self.fatal_reward = torch.rand(self.memory_size, 1)
        self.fatal_solution = [True for i in range(self.memory_size)]
        
        self.tar_model = tar_model
        self.model = act_model
        self.optimiser = optim.Adam(self.model.parameters(), lr = conf.memory.lr)

        # o for ordinary
        self.x_o_f = torch.rand(self.batch, num_features)
        self.x_o_u = torch.rand(self.batch, conf.gnn.p_dim)
        self.x_o_T = torch.rand(self.batch, conf.gnn.p_dim)
        self.x_o_k = torch.rand(self.batch, 1)
        self.y_o = torch.rand(self.batch, 1)
        
        # s for special
        self.x_s_f = torch.rand(self.batch, num_features)
        self.x_s_u = torch.rand(self.batch, conf.gnn.p_dim)
        self.x_s_T = torch.rand(self.batch, conf.gnn.p_dim)
        self.x_s_k = torch.rand(self.batch, 1)
        self.y_s = torch.rand(self.batch, 1)
        

    def memorise(self, memory, target, done, reward, solution, mode = 0):
        
        if mode == 0:
            self.data_x_u[self.num_data] = memory[0].view(1,-1)
            self.data_x_T[self.num_data] = memory[1].view(1,-1)
            self.data_x_k[self.num_data] = memory[2].view(1,-1)
            self.data_x_f[self.num_data] = memory[3].view(1,-1)

            self.data_y_u[self.num_data] = target[0]
            self.data_y_T[self.num_data] = target[1]
            self.data_y_k[self.num_data] = target[2]
            self.data_y_f[self.num_data] = target[3]
        
            self.data_done[self.num_data] = done
            self.data_reward[self.num_data] = reward
            self.data_solution[self.num_data] = solution
        
            self.num_data += 1
            self.max_num_data += 1
            if self.num_data >= self.memory_size: self.num_data = 0
                
        elif mode == 1:
            self.fatal_x_u[self.num_fatal] = memory[0].view(1,-1)
            self.fatal_x_T[self.num_fatal] = memory[1].view(1,-1)
            self.fatal_x_k[self.num_fatal] = memory[2].view(1,-1)
            self.fatal_x_f[self.num_fatal] = memory[3].view(1,-1)

            self.fatal_y_u[self.num_fatal] = target[0]
            self.fatal_y_T[self.num_fatal] = target[1]
            self.fatal_y_k[self.num_fatal] = target[2]
            self.fatal_y_f[self.num_fatal] = target[3]
        
            self.fatal_done[self.num_fatal] = done
            self.fatal_reward[self.num_fatal] = reward
            self.fatal_solution[self.num_fatal] = solution
        
            self.num_fatal += 1
            self.max_num_fatal += 1
            if self.num_fatal >= self.memory_size: self.num_fatal = 0
            
     
    def play(self):
        
        # Just for detection if to train =====================================
        if conf.memory.best_memory_play == True:
            if self.num_data < self.batch or self.num_fatal < self.batch: return 
        else:
            if self.num_data < self.collect_steps: return  
            
        if self.max_num_data >= self.memory_size: self.max_num_data = self.memory_size
        if self.max_num_fatal >= self.memory_size: self.max_num_fatal = self.memory_size
                
        if self.show:
            print('Model starts being trained...')
            self.show = False
            
        # Memory play contents ===============================================
        
        seed = int(abs(self.data_reward.mean())*100)
        random.seed(seed)
        
        sample_index1 = random.sample(range(self.max_num_data), self.batch)
        self.x_o_f[:self.batch,:] = self.data_x_f[sample_index1,:].detach()
        self.x_o_u[:self.batch,:] = self.data_x_u[sample_index1,:].detach()
        self.x_o_T[:self.batch,:] = self.data_x_T[sample_index1,:].detach()
        self.x_o_k[:self.batch,:] = self.data_x_k[sample_index1,:].detach() 
        x_o_done   = self.data_done[sample_index1,:]
        x_o_reward = self.data_reward[sample_index1,:]
        
        if conf.memory.best_memory_play == True:
            
            seed = int(abs(self.fatal_reward.mean())*100)
            random.seed(seed)
        
            sample_index2 = random.sample(range(self.max_num_fatal), self.batch)
            self.x_s_f[:self.batch,:] = self.fatal_x_f[sample_index2,:].detach()
            self.x_s_u[:self.batch,:] = self.fatal_x_u[sample_index2,:].detach()
            self.x_s_T[:self.batch,:] = self.fatal_x_T[sample_index2,:].detach()
            self.x_s_k[:self.batch,:] = self.fatal_x_k[sample_index2,:].detach()   
            x_s_done   = self.fatal_done[sample_index2,:]
            x_s_reward = self.fatal_reward[sample_index2,:]        

        for i in range(self.batch):
            seed = int(abs(self.data_reward.mean())*100) + i*int(abs(self.data_reward.max())*100)
            if x_o_done[i] == True: self.y_o[:self.batch,:] = x_o_reward[i]     
            elif x_o_done[i] == False:
                input1 = self.data_y_u[sample_index1[i]].detach()
                input2 = self.data_y_T[sample_index1[i]].detach()
                input3 = self.data_y_k[sample_index1[i]].detach()
                input4 = self.data_y_f[sample_index1[i]].detach() 
                
                input3 = torch.ones(input1.shape[0], 1) * input3
                input4 = torch.cat([input4,input4],dim = 0)
                
                state_val = self.tar_model([input1, input2, input3, input4])
                action_ = pick_action(int(input1.shape[0]/2), 0, 
                                     self.data_solution[sample_index1[i]], 
                                     state_val, seed)
                maxVal_ = state_val[action_].detach().clone()
                
                self.y_o[i] = (x_o_reward[i] + conf.rl.gamma*maxVal_).detach().clone()
                del input1, input2, input3, input4, seed
                del state_val, action_, maxVal_

        if conf.memory.best_memory_play == True:
            for i in range(self.batch):
                seed = int(abs(self.data_reward.mean())*100) + i*int(abs(self.data_reward.max())*100)
                if x_s_done[i] == True: self.y_s[:self.batch,:] = x_s_reward[i]     
                elif x_s_done[i] == False:
                    input1 = self.fatal_y_u[sample_index2[i]].detach()
                    input2 = self.fatal_y_T[sample_index2[i]].detach()
                    input3 = self.fatal_y_k[sample_index2[i]].detach()
                    input4 = self.fatal_y_f[sample_index2[i]].detach() 
                    
                    input3 = torch.ones(input1.shape[0], 1) * input3
                    input4 = torch.cat([input4,input4],dim = 0)
                    
                    state_val = self.tar_model([input1, input2, input3, input4])
                    action_ = pick_action(int(input1.shape[0]/2), 0, 
                                         self.fatal_solution[sample_index2[i]], 
                                         state_val, seed)
                    maxVal_ = state_val[action_].detach().clone()
                    
                    self.y_s[i] = (x_s_reward[i] + conf.rl.gamma*maxVal_).detach().clone()
                    del input1, input2, input3, input4, seed
                    del state_val, action_, maxVal_            


        
        output_o  = self.model([self.x_o_u.detach(), self.x_o_T.detach(), 
                                self.x_o_k.detach(), self.x_o_f.detach()])
        mse = self.loss(output_o, self.y_o.detach())
        
        if conf.memory.best_memory_play == True:
            output_s  = self.model([self.x_s_u.detach(), self.x_s_T.detach(), 
                                    self.x_s_k.detach(), self.x_s_f.detach()])       
            mse += self.loss(output_s, self.y_s.detach())    
            

        self.optimiser.zero_grad()
        mse.backward(retain_graph=True)         
        self.optimiser.step()  
    

        del output_o, mse, sample_index1
        if conf.memory.best_memory_play == True:
            del output_s, sample_index2