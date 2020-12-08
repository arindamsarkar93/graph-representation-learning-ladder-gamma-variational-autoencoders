from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import cPickle as pkl
import sys
from input_data import parse_index_file
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

"""
Boilerplate code used from https://github.com/tkipf/gcn
"""

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('embeds_path','','Path to file containing trained embeddings')
flags.DEFINE_string('dataset','', 'Dataset to evaluate on')
flags.DEFINE_integer('num_pca_components',0, 'No. of PCA components to use')
flags.DEFINE_integer('num_layers_to_use',1,'No. of embedding layers to use')

def one_hot2int(one_hot_rep):
    return [np.where(vec == 1)[0][0] for vec in one_hot_rep]

def train_test(train_embeds,labels_train, test_embeds, labels_test, val_embeds = None, labels_val = None):
  np.random.seed(1)
  from sklearn.linear_model import SGDClassifier
  from sklearn.metrics import f1_score

  #change from one-hot rep to integer labels
  labels_train = one_hot2int(labels_train)
  labels_test = one_hot2int(labels_test)
  labels_val = one_hot2int(labels_val)
  
  #logistic regression classifier
  clf = SGDClassifier(loss='log', class_weight = "balanced")

  #feature normalization?
  #check if sparse
  if  not sp.issparse(train_embeds): #False and
    scaler = StandardScaler()
    scaler.fit(train_embeds)
    train_embeds = scaler.transform(train_embeds)
    test_embeds = scaler.transform(test_embeds)
    val_embeds = scaler.transform(val_embeds)

    #PCA?
    if FLAGS.num_pca_components > 0:
        print ("Using PCA with ",FLAGS.num_pca_components," components")
        pca = PCA(n_components = FLAGS.num_pca_components)
        pca.fit(train_embeds)

        train_embeds = pca.transform(train_embeds)
        test_embeds = pca.transform(test_embeds)
        val_embeds = pca.transform(val_embeds)

  clf.fit(train_embeds, labels_train)
  predictions = clf.predict(test_embeds)

  accuracy = np.sum (predictions == labels_test)/len(labels_test)

  print("Accuracy: ", accuracy)

  #F1 score
  f1 = 0.0

  f1 = f1_score(labels_test, predictions, average="weighted")
  print("F1 score (weighted): ", f1)
  
def load_labels(dataset):
    # load the data: x, tx, allx, graph
    names = ['x','tx', 'y', 'ty', 'allx', 'ally']
    objects = []
    for i in range(len(names)):
        filename = "../data/ind.{}.{}".format(dataset, names[i])
        #print filename
        with open(filename, 'rb') as f:
            if sys.version_info > (3,0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, y, ty, allx, ally = tuple(objects)

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


    features = sp.vstack((allx, tx)).tolil() # convert to linked list
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    """
    The indices of test instances in graph for the transductive setting are 
    from #x to #x + #tx - 1, with the same order as in tx
    """


    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    print ("Test samples: ", len(idx_test), "Train samples: ", len(idx_train), "Val samples: ", len(idx_val))

    train_labels = labels[idx_train]
    test_labels = labels[idx_test]
    val_labels = labels[idx_val]

    return idx_train, idx_test, idx_val, labels, features

# Load embeds, concatenate embeddings
def load_embeds(embeds_path, num_concat=1):
    data = np.load(embeds_path)

    thetas = data['theta']
    num_embed_layers = len(thetas)
    print ("No. of embedding layers: ", num_embed_layers)

    thetas_ = [np.array([np.array(node_rep) for node_rep in thetas[i]]) for i in range(num_concat)]
    
    theta_concat = np.concatenate(thetas_, axis=1)

    print ("Final embedding shape: ", theta_concat.shape)
    return theta_concat

def main():
    num_layers_to_use = FLAGS.num_layers_to_use 
    print ("Dataset details: ", FLAGS.embeds_path)
    print ("No. of layers used (embeddings): ", num_layers_to_use)
    embeds = load_embeds(FLAGS.embeds_path, num_layers_to_use)
    idx_train, idx_test, idx_val, labels, features = load_labels(FLAGS.dataset)
   
    #size consistency check
    assert embeds.shape[0] == labels.shape[0]

    #Prepare train, test embeds
    train_embeds = embeds[idx_train]
    test_embeds = embeds[idx_test]
    val_embeds = embeds[idx_val]
    
    #Prepare labels
    train_labels = labels[idx_train]
    test_labels = labels[idx_test]
    val_labels = labels[idx_val]

    #Prepare actual features
    train_feats = features[idx_train]
    test_feats = features[idx_test]
    val_feats = features[idx_val]
    
    print("\n")
    
    print ("Training on learned embeddings:")
    train_test(train_embeds, train_labels, test_embeds, test_labels, val_embeds, val_labels)
    
    print("\n")

    print ("Training on actual features:")
    train_test(train_feats, train_labels, test_feats, test_labels, val_feats, val_labels)
    
    print ("\n\n")
if __name__ == '__main__':
    main()

