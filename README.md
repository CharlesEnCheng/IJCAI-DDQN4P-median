# IJCAI-DDQN4P-median

## Steps
### Step1
(1) config.py defines data path, model path, log path, d_dim, and U~(c,d) with mode 0\1

### Step2 
(1) instance_generator.py is required to be run in advance to create synthetic data.
(2) Xpress for optimisation should be installed beforehand; or commment the lines <from xpress_ver0417 import get_format, get_solution_pmedian> and its related lines in the below .pys)

    -feature.py 
    -gnn.py 
    -concat.py 


### Step3
replace c, d, mode in the conf to run the 3 different experiments and run 

    -feature.py 
    -gnn.py 
    -concat.py 
    
    
___

## Code notes

### utils.py
*(1) Common Functions - used by all three methods*

    '**Log**: input <feature/gnn/concat>; record vlidation results during training. '
    -**get_graphs**: input <train/test/valid>; return all files matching c,d,mode in the folder. 
    -**normalise**: input features(node representations); return normalised features. if a feature column's std = 0, skip the column.
    -**summary**: input validation results; return the summaried results and log texts.


*(2) Feature Related Functions  - used by feature.py & concat.py*

    -get_features: input n:number of facilities, k, instance; return the features of the facilities of a graph. This is only run at the initial state. 
    -update_features: input n, instance, left, feature; return the updated features. *Note*: *left* is a list returned by *get_left_customer()*, which yields a list showing that which customers haven't been supplied.


*(3) Gnn Related Functions  - used by gnn.py & concat.py*

    -agment: input instance, n; return another format of the instance for gnn related computations.
    -get_adj_matrix: input instance (returned from *agment()*); return the adjacent matrix of an instance.
    -get_dist_info: input instance (returned from *agment()*), weights\_model; return the sum of embedded weights of all incident edges to a node. As such, if there are 5 facilities and 5 customers, the returned matrix will be R<sub>10\*d</sub> *Note*: *weights_model* is a torch model created from *get_embed_weights(nn.Module)* defined in the three approaches, respectively.
    -important: *embed* is to do the embedding for T times, and  *get_values* is to get the value of a node<u>
    -embed: input embed_x, embed_u, embed_model, adj_mat, edge_mat; return updated embed_u, sum_nodes (sum of the updated embed_u). *Note*: *embed_x* is a list of if a node is picked/supplied, R<sub>2n</sub>. *embed_u* is the embedding, R<sub>2n\*d</sub>. *embed_model* defined in the 3 approaches, adj_mat returned from *get_adj_matrix()* and edge_mat from *get_dist_info()*. 
    -get_values: input embed_x, embed_u, embed_model, value_model, adj_mat, edge_mat, k; return embed_u, value_tensor, sum_nodes. *Note*: *embed_u* is the updated one, sum_nodes is the same as *embed()*'s return, and value_tensor is the predicted value of each node.


*(4) RL Related Functions*
get_left_customer: get customers that haven't been supplied by any picked facilities.
stop**: stop condition
**pick_action**: given number of n (n facilities/customers) , current solution and values of facilitiy nodes, return an action . 
**get_cost**: for those customers that have been served, pick the lowest weight from their picked suppliers/facilities.
    

*(5) tuned-parameter gnn functions - not used in this version but if gonna run tuned-parameter version, these functions will be used.*

**MySpMM**: define the return gradients of backpropogation in torch.
**gnn_spmm**: it calls MySpMM. This is to multiply the embeddings and the adjacent matrix. 
