#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 23:56:05 2021

@author: en-chengchang
"""

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  

from xpress_ver0417 import get_format, get_solution_pmedian
from utils import get_graphs
from config import conf
import numpy as np
from instance_utils import Data
import pickle

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== #  

valid_graphs = sorted(get_graphs('valid/'))
test_graphs  = sorted(get_graphs('test/'))

validation = {}

for graph in range(len(valid_graphs)):

    n = int(valid_graphs[graph].split('_')[4])
    k = int(valid_graphs[graph].split('_')[5].split('.')[0])
    instance = np.load(conf.path.data + 'valid/' + valid_graphs[graph])
    data = Data(instance, n, k)
    
    _ = data.augment()
    _ = data.get_adj_matrix()

    c = get_format(np.array(instance), n)    
    obj, _y, _x = get_solution_pmedian(c, n, k)    
    
    validation[valid_graphs[graph]] = -obj


c, d = conf.general.c, conf.general.d
a_file = open(conf.path.data + "valid/solution{0:03}-{1:03}.pkl".format(c,d), "wb")
pickle.dump(validation, a_file)
a_file.close()


test = {}

for graph in range(len(test_graphs)):

    n = int(test_graphs[graph].split('_')[4])
    k = int(test_graphs[graph].split('_')[5].split('.')[0])
    instance = np.load(conf.path.data + 'test/' + test_graphs[graph])
    data = Data(instance, n, k)
    
    _ = data.augment()
    _ = data.get_adj_matrix()

    c = get_format(np.array(instance), n)    
    obj, _y, _x = get_solution_pmedian(c, n, k)    
    
    test[test_graphs[graph]] = -obj
c, d = conf.general.c, conf.general.d
a_file = open(conf.path.data + "test/solution{0:03}-{1:03}.pkl".format(c,d), "wb")
pickle.dump(test, a_file)
a_file.close()
