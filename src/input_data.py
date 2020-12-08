import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data_semisup(dataset):

    """
    if dataset.startswith('llawyers'):
        adj, features, feature_presence = load_lawyers_mat(dataset)
        return adj, features, feature_presence
    elif dataset.startswith('yeast'):
        adj, features, feature_presence = load_yeast_mat(dataset)
        return adj, features, feature_presence
    elif dataset.startswith('nips12'):
        adj, features, feature_presence = load_nips_mat(dataset)
        return adj, features, feature_presence
    elif dataset.startswith('nips234'):
        adj, features, feature_presence = load_nips234_mat(dataset)
        return adj, features, feature_presence
    elif dataset.startswith('protein230'):
        adj, features, feature_presence = load_protein_mat(dataset)
        return adj, features, feature_presence
    """

    # load the data: x, tx, allx, graph
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("../data/ind.{}.{}".format(dataset, names[i]))))
    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]


    features = sp.vstack((allx, tx)).tolil() # convert to linked list
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    
    return y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data(dataset):

    if dataset.startswith('llawyers'):
        adj, features, feature_presence = load_lawyers_mat(dataset)
        return adj, features, feature_presence
    elif dataset.startswith('yeast'):
        adj, features, feature_presence = load_yeast_mat(dataset)
        return adj, features, feature_presence
    elif dataset.startswith('nips12'):
        adj, features, feature_presence = load_nips_mat(dataset)
        return adj, features, feature_presence
    elif dataset.startswith('nips234'):
        adj, features, feature_presence = load_nips234_mat(dataset)
        return adj, features, feature_presence
    elif dataset.startswith('protein230'):
        adj, features, feature_presence = load_protein_mat(dataset)
        return adj, features, feature_presence
    
    elif dataset.startswith('20ng'):
        adj, features, feature_presence = load_20ng(dataset)
        return adj, features, feature_presence
    
    elif dataset.startswith('text8'):
        adj, features, feature_presence = load_text8(dataset)
        return adj, features, feature_presence

    elif dataset.startswith('synthetic'):
        adj, features, feature_presence = create_synthetic_data()
        return adj, features, feature_presence
    elif dataset.startswith('2synthetic'):
        adj, features, feature_presence = create_synthetic_data2()
        return adj, features, feature_presence
    elif dataset.startswith('inv_synthetic'):
        adj, features, feature_presence = create_inv_synthetic_data()
        return adj, features, feature_presence
    elif dataset.startswith('overlapping_synthetic'):
        adj, features, feature_presence = create_overlapping_synthetic_data()
        return adj, features, feature_presence
    elif dataset.startswith('2overlapping_synthetic'):
        adj, features, feature_presence = create_overlapping_synthetic_data2()
        return adj, features, feature_presence
    
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("../data/ind.{}.{}".format(dataset, names[i]))))
    x, tx, allx, graph = tuple(objects)

    test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil() # convert to linked list
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features, 1

def load_lawyers_mat(dataset):

    mat_cont = sio.loadmat('../data/lazega-lawyers.mat')
    #mat_cont.keys()
    #>>>['A_adv', 'F_orig', '__globals__', 'F', 'A_work', '__header__', '__version__', 'A_friend']

    features = mat_cont['F']
    
    if dataset == 'llawyers_adv':
        adj = mat_cont['A_adv']
    elif dataset == 'llawyers_co-work':
        adj = mat_cont['A_work']
    elif dataset == 'llawyers_friends':
        adj = mat_cont['A_friend']

    adj = sp.csr_matrix(adj)
    features = sp.lil_matrix(features)
    
    return adj, features, 1
    
def load_yeast_mat(dataset):

    mat_data = sio.loadmat('../data/yeast.mat')

    #print mat_data.keys()
    #['B', '__version__', '__header__', '__globals__']

    adj = mat_data['B']
    adj = sp.csr_matrix(adj)
    features = sp.identity((adj.shape[0]))

    return adj, features, 0

def load_protein_mat(dataset):

    mat_data = sio.loadmat('../data/Protein230.mat')
    print mat_data.keys()
    #['B', '__version__', '__header__', '__globals__']
    
    adj = mat_data['B']
    adj = sp.csr_matrix(adj)
    features = sp.identity((adj.shape[0]))

    return adj, features, 0

def load_20ng(dataset):

    from word2vec_functions import word2vec_feats
    
    adj = sp.load_npz('../data/20ng/20ng.npz')
    vocab_path = '../data/20ng/vocab.npy' 
    word2vec_path = '/data1/arindam/gnews_w2v_300.bin'
    
    features = sp.csr_matrix(word2vec_feats(vocab_path, word2vec_path))

    #features = sp.identity((adj.shape[0]))

    return adj, features, 0
    
def load_text8(dataset):

    adj = sp.load_npz('../data/text8/text8.npz')

    features = sp.identity((adj.shape[0]))

    return adj, features, 0

def load_nips234_mat(dataset):

    mat_data = sio.loadmat('../data/nips234.mat')

    print mat_data.keys()
    #['B', '__version__', '__header__', '__globals__']
    
    adj = mat_data['B']
    adj = sp.csr_matrix(adj)
    features = sp.identity((adj.shape[0]))

    return adj, features, 0
    
def load_nips_mat(dataset):

    mat_data = sio.loadmat('../data/nips12.mat')
    # print mat_data.keys()

    adj = mat_data['B']
    adj = sp.csr_matrix(adj)
    features = sp.identity((adj.shape[0]))

    return adj, features, 0    

def load_data_split(dataset_str, split_idx):
    
    data_path = '../data/all_edge_idx_' + dataset_str + '.npy'
    all_edge_idx_array = np.load(data_path)

    return all_edge_idx_array[split_idx]

def load_masked_test_edges_for_kfold(dataset_str, k_fold=5, split_idx=0):

    data_path = '../data/' + dataset_str + '/' + str(k_fold) + '-fold/split_' + str(split_idx) + '.npz'
    data = np.load(data_path)

    return data['k_adj_train'], data['k_train_edges'], data['k_val_edges'], data['k_val_edges_false'], data['test_edges'], data['test_edges_false']

def load_masked_train_edges(dataset_str, split_idx=0):

    data_path = '../data/' + dataset_str + '/tr_split_' + str(split_idx) + '.npz'
    data = np.load(data_path)
    
    return data['adj_train'], data['train_edges'], data['train_edges_false']
    
def load_masked_test_edges(dataset_str, split_idx=0):

    data_path = '../data/' + dataset_str + '/split_' + str(split_idx) + '.npz'
    data = np.load(data_path)
    
    return data['adj_train'], data['train_edges'], data['val_edges'], data['val_edges_false'], data['test_edges'], data['test_edges_false']
    
def create_synthetic_data(N = 150, comm = 10):

    assert N % comm == 0, 'Num Communities should be a factor of N'
    nodes_per_comm = N / comm
    
    adj = np.zeros((N,N))
    
    for i in range(comm):
        adj[i*nodes_per_comm:(i+1)*nodes_per_comm, i*nodes_per_comm:(i+1)*nodes_per_comm] = 1
            
    adj = sp.csr_matrix(adj)
    features = sp.identity((adj.shape[0]))
        
    return adj, features, 0

def create_synthetic_data2(N = 150, comm = 10):

    assert N % comm == 0, 'Num Communities should be a factor of N'
    nodes_per_comm = N / comm
    
    f = np.zeros((N,comm))
    
    for i in range(comm):
        f[i*nodes_per_comm:(i+1)*nodes_per_comm, i] = 1
            
    D = np.random.normal(0,1,[comm,comm])
    W = (D + np.transpose(D))/2
    
    adj = np.round(sigmoid(np.matmul(np.matmul(f, W), np.transpose(f))))

    adj = sp.csr_matrix(adj)
    features = sp.identity((adj.shape[0]))
        
    return adj, features, 0

def create_inv_synthetic_data(N = 150, comm = 10):

    assert N % comm == 0, 'Num Communities should be a factor of N'
    nodes_per_comm = N / comm
    
    adj = np.ones((N,N))
    
    for i in range(comm):
        adj[i*nodes_per_comm:(i+1)*nodes_per_comm, i*nodes_per_comm:(i+1)*nodes_per_comm] = 0
            
    adj = sp.csr_matrix(adj)
    features = sp.identity((adj.shape[0]))
        
    return adj, features, 0

def create_overlapping_synthetic_data(N = 150, comm = 5):

    assert N % comm == 0, 'Num Communities should be a factor of N'
    nodes_per_comm = N / comm
    
    adj = np.zeros((N,N))
    
    for i in range(comm):
        adj[i*nodes_per_comm:(i+1)*nodes_per_comm, i*nodes_per_comm:(i+1)*nodes_per_comm] = 1

    adj[45:60, 0:30] = 1
    adj[45:60, 120:] = 1
    adj[0:30, 45:60] = 1
    adj[120:, 45:60] = 1
        
    adj = sp.csr_matrix(adj)
    features = sp.identity((adj.shape[0]))

    return adj, features, 0

def create_overlapping_synthetic_data2(N = 150, comm = 5):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x)) 
    
    n = 100
    z = 10
    comms = 10


    nz = np.zeros((n,z))

    nz[0:20, 0] = 1
    nz[20:35, 1] = 1
    nz[35:50, 2] = 1
    nz[50:65, 3] = 1
    nz[65:75, 4] = 1
    nz[75:85, 5] = 1
    nz[85:90, 6] = 1
    nz[90:95, 7] = 1
    nz[90:95, 9] = 1
    nz[95:100, 8] = 1

    # overlap
    nz[25:60, 7] = 1
    nz[60:80, 8] = 1
    W = np.eye(z)
    adj = sigmoid(np.matmul(np.matmul(nz, W), np.transpose(nz)))
    adj = np.round(adj)
    
    adj = sp.csr_matrix(adj)
    features = sp.identity((adj.shape[0]))

    return adj, features, 0
