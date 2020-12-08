import time
import os
import sys

import numpy as np
import scipy.sparse as sp

from input_data import load_data
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges, mask_test_edges_for_kfold

dataset_str = 'nips12'
prop_train = np.arange(15,90,5) #jumps of 5


#data_dir = 'data/all_edge_idx_' + dataset_str

# Load data. Raw adj is NxN Matrix and Features is NxF Matrix. Using sparse matrices here (See scipy docs). 
adj, features, _ = load_data(dataset_str)

# Store original adjacency matrix (without diagonal entries) for later
print ("Adj Original Matrix: " + str(adj.shape))

k_fold = False

if k_fold:
        out_str = 'data/' + dataset_str + "/missing/"
        if not os.path.exists(out_str):
                os.makedirs(out_str)

        for i in range(len(prop_train)):
                train = prop_train[i]
                val = 5.
                test = 100. - train - val

                k_adj_train, k_train_edges, k_val_edges, k_val_edges_false, test_edges, test_edges_false = mask_test_edges_for_kfold(adj, 5)

                np.savez(out_str+"split_" + str(i), k_adj_train=np.asarray(k_adj_train), k_train_edges=np.asarray(k_train_edges),
                         k_val_edges=np.asarray(k_val_edges), k_val_edges_false=np.asarray(k_val_edges_false),
                         test_edges=np.asarray(test_edges), test_edges_false=np.asarray(test_edges_false))

else:
        
        out_str = '../data/' + dataset_str + "/missing/"
        if not os.path.exists(out_str):
                os.makedirs(out_str)

        for i in range(len(prop_train)):
                train = prop_train[i]
                val = 5.
                test = 100. - train - val

                adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj, test_split = test, val_split = val,  all_edge_idx = None)

                # making array to save in file
                adj_train_a = []
                adj_train_a.append(adj_train)
                np.savez(out_str+"split_" + str(i), adj_train=np.asarray(adj_train_a), train_edges=np.asarray(train_edges),
                         val_edges=np.asarray(val_edges), val_edges_false=np.asarray(val_edges_false),
                         test_edges=np.asarray(test_edges), test_edges_false=np.asarray(test_edges_false))
    
