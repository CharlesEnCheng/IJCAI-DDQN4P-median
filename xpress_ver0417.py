

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
from config import conf
os.environ["XPRESS"] = conf.path.xpress

import xpress as xp
xp.controls.outputlog = 0
import numpy as np


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


