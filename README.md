# IJCAI-DDQN4P-median

config.py defines data path, model path, log path, d_dim, and U~(c,d) with mode 0\1

before training, 
(1) instance_generator.py is required to be run in advanced to get data.
(2) Xpress for optimisation is installed beforehand. (or commment the optimisation process in the below codes)

feature.py trains/produces the feature-based model
gnn.py trains/produces the fixed-parameter GNN model
concat.py trains/produces the hybrid model

