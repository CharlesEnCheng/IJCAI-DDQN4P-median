#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:21:13 2021

@author: en-chengchang
"""

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  

import torch
from torch import nn
from torch.autograd import Variable
from config import conf

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

class WeightsModel(nn.Module):
    def __init__(self, p_dim, seed = conf.general.seed):
        super().__init__()
        torch.manual_seed(seed)      
        self.w_in = nn.Linear(1, p_dim) 

    def forward(self, w):
        return self.w_in(w)

class EmbeddingModel(nn.Module):
    def __init__(self, x_in, p_dim, seed = conf.general.seed, T = conf.gnn.T, feature_num = None):
        super().__init__()
        torch.manual_seed(seed)
        self.T = T
        
        self.x_in = nn.Linear(x_in, p_dim) 
        if feature_num == None:
            self.u_in = nn.Linear(p_dim, p_dim) 
        else:
            self.u_in = nn.Linear(feature_num, p_dim) 
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

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  
  
if conf.general.feature == True and conf.general.gnn == True and\
    conf.general.concat == True:
        
    class ValueModel(nn.Module):
        def __init__(self, p_dim, feature_num, seed = conf.general.seed):
            super().__init__()
            torch.manual_seed(seed)

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

    
            output = self.output(torch.tanh(layer))
            output = self.output2(output)
            return output
        
elif (conf.general.feature == True and conf.general.gnn == True and\
    conf.general.concat == False) or (conf.general.feature == False and\
    conf.general.gnn == True):     
        
    class ValueModel(nn.Module):
        def __init__(self, p_dim, feature_num, seed = conf.general.seed):
            super().__init__()
            torch.manual_seed(seed)

            self.node_in = nn.Linear(p_dim, 3*p_dim) 
            self.sumn_in = nn.Linear(p_dim, 2*p_dim) 
            self.k_in = nn.Linear(1, 2*p_dim) 
            self.output = nn.Linear(7*p_dim, 2*p_dim) 
            self.output2 = nn.Linear(2*p_dim, 1) 
            
        def forward(self, features):
            node = features[0] 
            sum_node = features[1] 
            k = features[2] 

            node_layer = torch.tanh(self.node_in(node))
            sum_layer  = torch.tanh(self.sumn_in(sum_node))
            k_layer    = torch.tanh(self.k_in(k))
            layer = torch.cat((node_layer, sum_layer, k_layer), dim=1)

            output = self.output(torch.tanh(layer))
            output = self.output2(output)
            return output

elif conf.general.feature == True and conf.general.gnn == False:                
        
    class ValueModel(nn.Module):
        def __init__(self, p_dim, feature_num, seed = conf.general.seed):
            super().__init__()
            torch.manual_seed(seed)
            self.layer1 = nn.Linear(feature_num, feature_num * 2) 
            self.layer2 = nn.Linear(feature_num * 2, feature_num * 6) 
            self.layer3 = nn.Linear(feature_num * 6, feature_num * 3) 
            self.output = nn.Linear(feature_num * 3, 1) 
    
        def forward(self, features):
            value = torch.tanh(self.layer1(features[3]))
            value = torch.relu(self.layer2(value))
            value = torch.relu(self.layer3(value))
            value = self.output(value)
            return value
