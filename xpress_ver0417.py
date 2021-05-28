

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:01:04 2021

@author: en-chengchang
"""

# =========================================================================== #
#                                                                             #
#                                                                             #
#                                                                             #
# =========================================================================== #

import os
os.environ["XPRESS"] = "/Users/en-chengchang/Desktop/FICO/"

import xpress as xp
xp.controls.outputlog = 0
from numpy import genfromtxt
import numpy as np

import matplotlib.pyplot as plt
import torch

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== # 

def distance(node0: list, node1: list):
    return ((node0[0] - node1[0]) ** 2 + (node0[1] - node1[1]) ** 2)**0.5
    
def get_instance(n: int, k: int, plot = False, distribution = 'chi',
                graph = 'k-median', seed = None):
    assert distribution in ['chi', 'uni']
    assert graph in ['k-median', 'hub']
    
    if seed != None:
        np.random.seed(seed)
        
    instance = []
    instance.append([n, k])
    
    if distribution == 'chi':
        x = np.random.chisquare(3, 2*n)
        y = np.random.chisquare(4, 2*n)

    if distribution == 'uni':
        x = np.random.chisquare(3, 2*n)
        y = np.random.chisquare(4, 2*n)
    
    x =  ((x-min(x))/(max(x)-min(x))) * 10
    y =  ((y-min(y))/(max(y)-min(y))) * 10
    
    for i in range(len(x)): # looping nodes
        if plot:
            if i < n: color = 'ro' # for facility
            else    : color = 'bo' # for customer
            if graph == 'hub': color = 'bo'
            plt.plot(x[i], y[i], color) 
        for j in range(len(x)):
            instance.append([i, j, distance([x[i],y[i]], [x[j],y[j]])])
            if plot:
                plt.plot([x[i],x[j]],[y[i],y[j]], 'g--', alpha = 0.1) 
    if plot: plt.show()
    return instance

def get_adj_matrix(instance: list):
    n = instance[0][0] * 2
    adj_mat = torch.zeros(n, n)
    for i, j, val in instance[1:]:
        adj_mat[i][j] = 1
    adj_mat = adj_mat
    return adj_mat

# =========================================================================== #
#                                                                             #
#                                                                             #
#                                                                             #
# =========================================================================== #

def get_format(x, n):

    c = np.zeros([n,n]) + (10 ** 7 - 1)
    for row in range(len(x)):
        c[int(x[row,1]), int(x[row,0])] = x[row,2]
        # row is for customers; column is for facilties
    return c

def get_solution_pmedian(c, n, k):
    x = np.array ([xp.var (vartype = xp.binary) for i in range(n*n)]).reshape(n,n)
    y = np.array([xp.var (vartype = xp.binary) for i in range(n)])

    c1 = [xp.Sum(y) <= k]
    c2 = [xp.Sum(x[i,:]) >= 1 for i in range(n)]
    c3 = [y[j] - x[i,j] >= 0 for i in range(n) for j in range(n)]

    objective = xp.Sum(xp.Sum(c * x))

    p = xp.problem()
    p.addVariable(x,y)
    p.setObjective (objective, sense = xp.minimize)
    p.addConstraint(c1, c2, c3)
    p.solve()

    y_solu = p.getSolution(y)
    x_solu = p.getSolution(x)
    
    p.getSolution(objective)
   
    return p.getSolution(objective), y_solu, x_solu


def get_mat(instance_s, n):
    mat = np.zeros([n*2,n*2]) + (10 ** 7 - 1)
    for i,j,m in instance_s[1:]:
        mat[i, j] = m
        mat[j, i] = m
    return mat


def get_solution_hlp(mat, k, wij = 1, alpha = 0):    
    r = mat.shape[0]
    x = np.array ([xp.var (vartype = xp.binary) for i in range(r*r)]).reshape(r,r)
    c1 = [xp.Sum(x[i,:]) == 1 for i in range(r)] 
    c2 = [xp.Sum([x[i,i] for i in range(r)]) == k]
    c3 = [x[i,i] - x[j,i] >= 0 for i in range(r) for j in range(r) if i != j]
    
    ref = mat * x
    objective = 0
    
    for i in range(r):
        for j in range(r):
            a = xp.Sum(ref[i])
            b = xp.Sum(ref[j])
            
            for k in range(r):
                for m in range(r):
                    c = alpha*x[i,k]*x[j,m]*mat[k,m]
            objective += wij*(a+b+c)
    
    
    p = xp.problem()
    p.addVariable(x)
    p.setObjective (objective/2, sense = xp.minimize)
    p.addConstraint(c1, c2, c3)
    p.solve()

    x_solu = p.getSolution(x)
    
    p.getSolution(objective)
   
    return p.getSolution(objective), x_solu




# =========================================================================== #
#                                                                             #
#                                                                             #
#                                                                             #
# =========================================================================== #
if __name__ == '__main__':

    for seed in [0]:
         n = 35
         k = 15
         
         if seed <5: typr = 'chi'
         else: typr = 'uni'
         instance_s = get_instance(n, k, seed = seed, distribution = typr)
         
         instance = [i for i in instance_s if i[0] < n and i[1] >= n]
         instance = [[i,j-n,k] for i,j,k in instance]
         instance = np.array(instance)
         
         
         mat = get_mat(instance_s, n)
         
         
         c = get_format(instance, n)
    
         obj, y_mat ,x_mat = get_solution_pmedian(c, n, k)
         
         hlp_obj, x_solu = get_solution_hlp(mat, k, wij = 1, alpha = 0)
         
    
    
     
    
    y = np.where(y_mat == 1)[0]
    cost = 0
    for index in range(n):
        tmp = []
        for row in instance:
            if row[1] == index and row[0] in y:
                tmp.append(row[2])
        cost += min(tmp)
        
         
    wij = 1
    alpha = 0
    x = x_solu
    r = 2*n
    
    np.diagonal(x).sum()
    for i in range(r):
        if x[i,:].sum() != 1:
            print(x[i,:].sum())
    for i in range(r):
        for j in range(r):
            if x[i,i] - x[j,i] < 0:
                print(x[i,i], x[j,i])
                
    np.where(np.diagonal(x)==1)