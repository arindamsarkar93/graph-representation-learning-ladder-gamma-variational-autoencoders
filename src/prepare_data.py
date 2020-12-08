import time
import os

import numpy as np
import scipy.sparse as sp

from input_data import load_data
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges, mask_test_edges_for_kfold

dataset_str = 'synthetic2'

#data_dir = 'data/all_edge_idx_' + dataset_str

# Load data. Raw adj is NxN Matrix and Features is NxF Matrix. Using sparse matrices here (See scipy docs). 
adj, features, _ = load_data(dataset_str)

# Store original adjacency matrix (without diagonal entries) for later
print ("Adj Original Matrix: " + str(adj.shape))

k_fold = False

if k_fold:
        out_str = '../data/' + dataset_str + "/5-fold/"
        if not os.path.exists(out_str):
                os.makedirs(out_str)

        for i in range(10):

                k_adj_train, k_train_edges, k_val_edges, k_val_edges_false, test_edges, test_edges_false = mask_test_edges_for_kfold(adj, 5)

                np.savez(out_str+"split_" + str(i), k_adj_train=np.asarray(k_adj_train), k_train_edges=np.asarray(k_train_edges),
                         k_val_edges=np.asarray(k_val_edges), k_val_edges_false=np.asarray(k_val_edges_false),
                         test_edges=np.asarray(test_edges), test_edges_false=np.asarray(test_edges_false))

else:
        
        out_str = '../data/' + dataset_str + "/"
        if not os.path.exists(out_str):
                os.makedirs(out_str)

        for i in range(10):
                
                adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj, test_split = 10.0, val_split = 5.0,  all_edge_idx = None)

                # making array to save in file
                adj_train_a = []
                adj_train_a.append(adj_train)
                np.savez(out_str+"split_" + str(i), adj_train=np.asarray(adj_train_a), train_edges=np.asarray(train_edges),
                         val_edges=np.asarray(val_edges), val_edges_false=np.asarray(val_edges_false),
                         test_edges=np.asarray(test_edges), test_edges_false=np.asarray(test_edges_false))
    
