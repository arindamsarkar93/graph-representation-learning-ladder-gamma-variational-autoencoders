#!/usr/bin/python
"""
Mostly similiar to train.py, except for the fact, that it is intended to be used for semisupervised setting, where all of the Adjacency matrix is used for training.
"""

from __future__ import division
from __future__ import print_function

import time
import os
import sys


import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

from optimizer import Optimizer
from input_data import load_data, load_masked_test_edges, load_masked_train_edges, load_masked_test_edges_for_kfold, load_data_semisup
from model import LadderGammaVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges, mask_train_edges

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Max number of epochs to train. Training may stop early if validation-error is not decreasing')
flags.DEFINE_string('hidden', '64_32', 'Number of units in hidden layers')

flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('model', 'lg_vae', 'Model to use')
flags.DEFINE_string('dataset', 'cora', 'Dataset string: cora, citeseer, pubmed, 20ng, llawyers_friends, llawyers_co-work, llawyers_adv, yeast, nips12, nips234, protein230')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

flags.DEFINE_float('alpha0', 10., 'Prior Alpha for Beta')

# Not using K-fold
flags.DEFINE_integer('use_k_fold', 0, 'Whether to use k-fold cross validation')
flags.DEFINE_integer('k', 5, 'how many folds for cross validation.')
flags.DEFINE_integer('save_pred_every', 5, 'Save summary after epochs')

flags.DEFINE_integer('early_stopping', 300, 'how many epochs to train after last best validation')

# Split to use for evaluation
flags.DEFINE_integer('split_idx', 0, 'Dataset split (Total:10) 0-9')
flags.DEFINE_integer('weighted_ce', 1, 'Weighted Cross Entropy: For class balance')
flags.DEFINE_integer('test', 0, 'Load model and run on test')

#options
flags.DEFINE_integer('use_kl_warmup', 0, 'Use a linearly increasing KL [0-1] coefficient -- see wu_beta in optimization.py')
flags.DEFINE_integer('use_x_warmup', 0, 'Use a linearly increasing [0-1] coefficient for multiplying with x_loss, annealing sort of -- see wu_x in optimization.py')

flags.DEFINE_float('bias_weight_1',1.0,'Multiplier for cross entropy loss for 1 label. See optimizer.py')
flags.DEFINE_string('expr_info','','Info about the experiment')
flags.DEFINE_float('lambda_mat_scale',0.1, 'Scale for Normal being used in initialization of scale parameter of lambda matrix')
flags.DEFINE_integer('cosine_norm',1,'Whether to use Cosine Normalized product instead of dot product for bilinear product')
flags.DEFINE_string('gpu_to_use','','Which GPU to use. Leave blank to use None')
flags.DEFINE_integer('reconstruct_x',0,'Whether to separately reconstruct x')
flags.DEFINE_integer('log_results',0,'Whether to log results')

#Use random_split for 20ng
flags.DEFINE_integer('random_split',1,'Whether to use random splits instead of fixed splits')
flags.DEFINE_integer('link_prediction',1,'Whether to add link prediction loss')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss.')

flags.DEFINE_integer('semisup_train',0,'Whether to perform semisupervised classification training as well')
flags.DEFINE_integer('mc_samples',1,'No. of MC samples for calculating gradients')

flags.DEFINE_string('data_type','binary','Type of data: binary, count')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = (FLAGS.gpu_to_use)

save_path_disk = "/data/lgvae/data_models"

graph_dir = save_path_disk + '/LG_VAE/' + dataset_str + '/'
save_dir =  save_path_disk + '/LG_VAE/' + dataset_str +'/split_'+ str(FLAGS.split_idx) + '/' + model_str + "/" + FLAGS.hidden + "/"
if not os.path.exists(save_dir):
        os.makedirs(save_dir)

k_fold_str = '_no-k-fold'
if FLAGS.use_k_fold:
    k_fold_str = str(FLAGS.k)+'-fold'

#Let's start time here
start_time = time.time()

#if dataset_str == "20ng":
#    assert FLAGS.random_split == 1

# Load data. Raw adj is NxN Matrix and Features is NxF Matrix. Using sparse matrices here (See scipy docs). 
adj, features, feature_presence = load_data(dataset_str)

if(FLAGS.semisup_train):
    y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_semisup(dataset_str)
    num_classes = y_train.shape[1]

#if(feature_presence == 0):
#   #save user from inadvertant errors
#   FLAGS.reconstruct_x = 0 

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

print ("Adj Original Matrix: " + str(adj_orig.shape))
print ("Features Shape: " + str(features.shape))
print (np.sum(adj_orig), " edges")

num_nodes = adj_orig.shape[0]
features_shape = features.shape[0]
if FLAGS.features == 0:
        features = sp.identity(features_shape)  # featureless

pos_weight_feats = float(features.shape[0] * features.shape[1] - features.sum()) / features.sum()
norm_feats = features.shape[0] * features.shape[1] / float((features.shape[0] * features.shape[1] - features.sum()) * 2) # (N+P) x (N+P) / (N)

# feature sparse matrix to tuples 
features = sparse_to_tuple(features.tocoo())     

def get_label_pred(sess, placeholders, feed_dict, model, S=5):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['is_training']: False})

    op = np.zeros([num_nodes, num_classes])
    if model_str == 'lg_vae':
        #get S posterior samples -> get S reconstructions
        for i in range(S): 
            outs = sess.run([model.z], feed_dict=feed_dict)
            op += outs[0]
            #adj_rec = adj_rec + outs[3]
    return op/S

def get_ll(sess, placeholders, feed_dict, opt, y_set, set_mask, S=5):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['is_training']: False})
    feed_dict.update({placeholders['node_labels']: y_set})
    feed_dict.update({placeholders['node_labels_mask']: set_mask})

    if(dataset_str == 'pubmed'):
        S=5
    else:
        S=15

    loss = 0.0

    if model_str == 'lg_vae':
        #get S posterior samples -> get S reconstructions
        for i in range(S): 
            outs = sess.run([opt.ae_loss], feed_dict=feed_dict)
            loss += outs[0]
            #adj_rec = adj_rec + outs[3]
    
    return loss/S


def get_score_matrix(sess, placeholders, feed_dict, model, S=1):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['is_training']: False})
        
    adj_rec = np.zeros([num_nodes, num_nodes]);

    if model_str == 'lg_vae':
        #get S posterior samples -> get S reconstructions
        for i in range(S): 
            outs = sess.run([model.reconstructions, model.poisson_rate], feed_dict=feed_dict)
            #print (outs[0])
            #print (outs[1])
            #outs_list.append(outs)
            #adj_rec, z_activated = monte_carlo_sample(outs[0], outs[1], outs[2], FLAGS.temp_post, S, sigmoid)
            #adj_rec = adj_rec + outs[3]

            adj_rec += np.reshape(outs[0], (num_nodes, num_nodes))

    #average
    adj_rec = adj_rec/S
    
    return adj_rec

"""
Get Semi-Supervised training accuracy on test set
"""
def get_semisup_acc(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = np.equal(np.argmax(preds, 1), np.argmax(labels, 1))
    accuracy_all = correct_prediction.astype(float)
    mask = mask.astype(float)
    mask /= np.mean(mask)
    accuracy_all *= mask
 
    return np.mean(accuracy_all)

"""
Get ROC score and average precision
"""
def get_roc_score(adj_rec, edges_pos, edges_neg, emb=None):
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    preds = []
    pos = []
    for e in edges_pos:
        #preds.append(sigmoid(adj_rec[e[0], e[1]]))
        preds.append(adj_rec[e[0], e[1]])
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        #preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        preds_neg.append(adj_rec[e[0], e[1]])
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    # Compute precision recall curve 
    precision, recall, _ = precision_recall_curve(labels_all, preds_all)
    
    auc_pr = auc(recall, precision)
    #auc_prm = auc_pr_m(preds_all, labels_all)
    #print (str(auc_pr))
    #print (str(auc_prm))
    #sys.exit()
    
    return roc_score, ap_score, auc_pr

def auc_pr_m(probs, true_labels):

        #prob_1 = probs*true_labels + (1 - probs)*(1 - true_labels)
        prob_1 = probs
        
        isort = np.argsort(-1*prob_1) # descend

        #[dummy, isort] = np.sort(prob_1, 'descend')
        precision = np.cumsum( true_labels[isort] ) / np.arange(1, len(prob_1)+1)
        recall    = np.cumsum( true_labels[isort] ) / np.sum( true_labels )

        print (type(recall))
        print (recall.shape)

        print (recall)
        print (precision)
        
        recall = np.insert(recall, 0, 0)
        precision = np.insert(precision, 0, 1)
        
        #area = trapz([0,recall],[1,precision]) %in matlab
        area = np.trapz(precision,recall)

        return area

# create_model 
def create_model(placeholders, adj, features):

    num_nodes = adj.shape[0]
    num_features = features[2][1]
    features_nonzero = features[1].shape[0] # Can be used for dropouts. See GraphConvolutionSparse
    
    #print(num_features)

    # Create model
    model = None
    if model_str == 'lg_vae':
        if FLAGS.semisup_train:
            model = LadderGammaVAE(placeholders, num_features, num_nodes, features_nonzero, num_classes, mc_samples = FLAGS.mc_samples)
        else:
            model = LadderGammaVAE(placeholders, num_features, num_nodes, features_nonzero, mc_samples = FLAGS.mc_samples)

    """
    if num_nodes > 10000:
        edges_for_loss = None
    else:
        edges_for_loss = placeholders['edges_for_loss']
    """
    edges_for_loss = placeholders['edges_for_loss']
    
    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'lg_vae':
            if FLAGS.semisup_train:
                opt = Optimizer(labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                            validate_indices=False), [-1]),
                                model=model, num_nodes=num_nodes,
                                pos_weight=placeholders['pos_weight'],
                                norm=placeholders['norm'],
				weighted_ce = FLAGS.weighted_ce,
                                edges_for_loss=edges_for_loss,
                                epoch=placeholders['epoch'],
                                features = tf.reshape(tf.sparse_tensor_to_dense(placeholders['features'], validate_indices = False), [-1]),
                                norm_feats = placeholders['norm_feats'],
                                pos_weight_feats = placeholders['pos_weight_feats'],
                                node_labels = placeholders['node_labels'],
                                node_labels_mask = placeholders['node_labels_mask'])
            else:
                opt = Optimizer(labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                            validate_indices=False), [-1]),
                                model=model, num_nodes=num_nodes,
                                pos_weight=placeholders['pos_weight'],
                                norm=placeholders['norm'],
				weighted_ce = FLAGS.weighted_ce,
                                edges_for_loss=edges_for_loss,
                                epoch=placeholders['epoch'],
                                features = tf.reshape(tf.sparse_tensor_to_dense(placeholders['features'], validate_indices = False), [-1]),
                                norm_feats = placeholders['norm_feats'],
                                pos_weight_feats = placeholders['pos_weight_feats'])


    return model, opt

def train(placeholders, model, opt, adj_train, train_edges, train_edges_false, features, sess, name="single_fold"):

    adj = adj_train
    
    # This will be calculated for every fold
    # pos_weight and norm should be tensors
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum() # N/P
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2) # (N+P) x (N+P) / (N)


    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Some preprocessing. adj_norm is D^(-1/2) x adj x D^(-1/2)
    adj_norm = preprocess_graph(adj)

    # get summaries
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))
    summary_op = tf.summary.merge(summaries)

    # initialize summary_writer
    summary_writer = tf.summary.FileWriter(graph_dir, sess.graph)
    meta_graph_def = tf.train.export_meta_graph(filename=graph_dir+'/model.meta')
    print("GRAPH IS  SAVED")
    sys.stdout.flush()
    
    # session initialize
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    #val_roc_score = []
    #best_validation = 0.0
    """
    num_nodes = adj.shape[0]
    if num_nodes < 10000:
        edges_for_loss = np.arange(num_nodes * num_nodes)
        ignore_edges = []
        edges_to_ignore = np.concatenate((val_edges, val_edges_false, test_edges, test_edges_false), axis=0)
        for e in edges_to_ignore:
                ignore_edges.append(e[0]*num_nodes+e[1])
        edges_for_loss = np.delete(edges_for_loss, ignore_edges, 0)
    else:
        edges_for_loss = []
    """

    edges_for_loss = np.ones((num_nodes * num_nodes), dtype = np.float32)
    ignore_edges = []
    edges_to_ignore = [] #np.concatenate((val_edges, val_edges_false, test_edges, test_edges_false), axis=0)
    for e in edges_to_ignore:
        ignore_edges.append(e[0] * num_nodes + e[1])

    edges_for_loss[ignore_edges] = 0.0

# Train model

    best_val_acc = 0.
    min_val_loss = 10e10
    for epoch in range(FLAGS.epochs):

        t = time.time()
        
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['is_training']: True})
        feed_dict.update({placeholders['norm']: norm})
        feed_dict.update({placeholders['pos_weight']: pos_weight})
        feed_dict.update({placeholders['edges_for_loss']: edges_for_loss})
        feed_dict.update({placeholders['epoch']: epoch})
        feed_dict.update({placeholders['norm_feats']: norm_feats})
        feed_dict.update({placeholders['pos_weight_feats']: pos_weight_feats})

        if(FLAGS.semisup_train):
            feed_dict.update({placeholders['node_labels']: y_train})
            feed_dict.update({placeholders['node_labels_mask']: train_mask})
        # Run single weight update

        """
        if epoch % FLAGS.save_pred_every == 0:
                outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.kl_term, model.reconstructions, model.clipped_logit, model.posterior_theta_params, model.shape_d, model.theta, opt.regularization, model.prior_theta_params, model.lambda_mat, summary_op], feed_dict=feed_dict)
                summary = outs[-1]
                summary_writer.add_summary(summary, epoch)
        else:

        """
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.kl_term, model.reconstructions, model.clipped_logit, model.posterior_theta_params, model.shape_d, model.theta, opt.regularization, model.prior_theta_params, model.lambda_mat, opt.grads_vars, opt.nll, opt.kl_term, opt.x_loss, opt.semisup_loss, opt.semisup_acc, model.z, opt.check, model.phi, opt.grads_vars], feed_dict=feed_dict)
        
        #outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.kl_term, model.reconstructions, model.clipped_logit, model.posterior_theta_params, model.shape_d, model.theta, opt.regularization, model.prior_theta_params, model.lambda_mat, opt.grads_vars, opt.nll, model.x_recon], feed_dict=feed_dict)
        
        #outs = sess.run([model.reconstructions, model.clipped_logit, model.posterior_theta_params, model.shape_d, model.theta, model.prior_theta_params, model.lambda_mat, model.x_recon], feed_dict=feed_dict)
        
        #print (outs[-1])
        #print (outs[-2])
        #print (outs[-3])
        #print (outs[-1].shape)
        # Compute average loss
        
        avg_cost = outs[1]
        avg_accuracy = outs[2]
        lambda_mat = outs[11]
        reconstructions = outs[4]
        clipped_logit = outs[5]
        posterior_theta_params = outs[6]
        shape_d = outs[7]
        theta = outs[8]
        regularization = outs[9]
        prior_theta_params = outs[10];
        nll = outs[13]
        x_loss = outs[15]
        semisup_loss = outs[16]
        semisup_acc = outs[17]
        model_z = outs[18]
        kl = outs[3]
        phi = outs[20]
        g_v = outs[21]

        #print (phi)
        #print(g_v)
        print (np.min(reconstructions), np.max(reconstructions), np.sum(reconstructions))
        if(np.isnan(kl)):
            print (posterior_theta_params)
            sys.exit()

        #print (avg_cost)
        print ('KL: ', kl, 'X_loss: ', x_loss, 'semisup_loss: ', semisup_loss, 'semisup train acc: ',semisup_acc, 'NLL: ', nll)
        #print (reconstructions)
        #print(outs[13])
        #print(outs[14])
        
               
        if True:#avg_accuracy > 0.9 or model_str == 'gcn_vae':

                #Validation
                #adj_rec = get_score_matrix(sess, placeholders, feed_dict, model, S=2)
                #roc_curr, ap_curr, _  = get_roc_score(adj_rec, val_edges, val_edges_false)
        
                model_z = get_label_pred(sess, placeholders, feed_dict, model, S=10)
                ss_val_acc = get_semisup_acc(model_z, y_val, val_mask)
                #ss_val_acc = 0.
                ss_val_loss = 0.
                #ss_val_loss = get_ll(sess, placeholders, feed_dict, opt, y_val, val_mask)
                
                print("Epoch:", '%03d' % (epoch + 1), "cost=", "{:.3f}".format(avg_cost), "kl=", "{:.3f}".format(outs[3]), "reg=", "{:.4f}".format(regularization), 
                      "val_acc=", "{:.3f}".format(ss_val_acc), #"val_roc=", "{:.3f}".format(roc_curr), "val_ap=", "{:.3f}".format(ap_curr), 
                      "time=", "{:.2f}".format(time.time() - t))
                print("\n")
                #roc_curr = round(roc_curr, 3)
                #val_roc_score.append(roc_curr)
            
                if ss_val_acc > best_val_acc:
                #if ss_val_loss < min_val_loss:
                    print("Saving model..")
                    saver.save(sess=sess, save_path=save_dir+name)
                    #best_sess = sess
                    best_val_acc = ss_val_acc
                    #min_val_loss = ss_val_loss
                    last_best_epoch = 0

                if last_best_epoch > FLAGS.early_stopping:
                        break;
                else:
                        last_best_epoch += 1
            
                """
                if roc_curr > best_validation:
                        # save model
                        print ('Saving model')
                        saver.save(sess=sess, save_path=save_dir+name)
                        best_validation = roc_curr
                        last_best_epoch = 0

                if last_best_epoch > FLAGS.early_stopping:
                        break;
                else:
                        last_best_epoch += 1
                """
                #else:
                #print("Training Epoch:", '%03d' % (epoch + 1), "cost=", "{:.3f}".format(avg_cost), #"reg=", "{:.1f}".format(regularization),
                #      "train_acc=", "{:.3f}".format(avg_accuracy), "time=", "{:.2f}".format(time.time() - t))

    print("Optimization Finished!")
    #val_max_index = np.argmax(val_roc_score)
    #print('Validation ROC Max: {:.3f} at Epoch: {:04d}'.format(val_roc_score[val_max_index], val_max_index))

    saver.restore(sess=sess, save_path=(save_dir+name))
    
    adj_score = get_score_matrix(sess, placeholders, feed_dict, model)
   
    #repeat for the sake of analysis
    roc_score, ap_score, auc_pr = get_roc_score(adj_score, train_edges, train_edges_false)
    
    #Use best val_acc session
    if(FLAGS.semisup_train):
        model_z = get_label_pred(sess, placeholders, feed_dict, model, S = 50)

    # Use this code for qualitative analysis
     
    qual_file = '../data/qual_' + dataset_str + '_' + model_str + k_fold_str + FLAGS.hidden
    theta_save = [layer_params.tolist() for layer_params in theta]

    # layer(s) --> nodes --> param(s)
    #Need to save phis too?
    posterior_theta_params_save = [[[node_params.tolist() for node_params in w_params] for w_params in layer_params] for layer_params in posterior_theta_params]
    np.savez(qual_file,theta = theta_save, posterior_theta_params = posterior_theta_params_save, lambda_mat = lambda_mat, roc_score = roc_score, ap_score = ap_score, auc_pr = auc_pr, expr_info = FLAGS.expr_info)       
    #saver.restore(sess=sess, save_path=(save_dir+name))

    return adj_score, model_z

def load_model(placeholders, model, opt, adj_train, train_edges, train_edges_false, features, sess, name="single_fold"):

        adj = adj_train
        # This will be calculated for every fold
        # pos_weight and norm should be tensors
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum() # N/P
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2) # (N+P) x (N+P) / (N)

        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        # Some preprocessing. adj_norm is D^(-1/2) x adj x D^(-1/2)
        adj_norm = preprocess_graph(adj)
    
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['is_training']: True})
        feed_dict.update({placeholders['norm']: norm})
        feed_dict.update({placeholders['pos_weight']: pos_weight})
        feed_dict.update({placeholders['norm_feats']: norm_feats})
        feed_dict.update({placeholders['pos_weight_feats']: pos_weight_feats})
        
        # Some preprocessing. adj_norm is D^(-1/2) x adj x D^(-1/2)
        adj_norm = preprocess_graph(adj)
        saver = tf.train.Saver()
        
        saver.restore(sess=sess, save_path=(save_dir+name))
        print ('Model restored')

        if (dataset_str == 'pubmed'): # decreasing samples. Num of nodes high
                S = 5
        else:
                S = 15
        
        adj_score = get_score_matrix(sess, placeholders, feed_dict, model, S=S)

        return adj_score

def main():

    num_nodes = adj_orig.shape[0]
    print ("Model is " + model_str)

    # Define placeholders
    if FLAGS.semisup_train:
        placeholders = {
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'is_training': tf.placeholder(tf.bool),
            'norm': tf.placeholder(tf.float32),
            'pos_weight': tf.placeholder(tf.float32),
            'edges_for_loss': tf.placeholder(tf.float32),
            'epoch': tf.placeholder(tf.int32),
            'norm_feats': tf.placeholder(tf.float32),
            'pos_weight_feats': tf.placeholder(tf.float32),
            'node_labels':  tf.placeholder(tf.float32),
            'node_labels_mask':  tf.placeholder(tf.int32)
        }
    else:
        placeholders = {
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'is_training': tf.placeholder(tf.bool),
            'norm': tf.placeholder(tf.float32),
            'pos_weight': tf.placeholder(tf.float32),
            'edges_for_loss': tf.placeholder(tf.float32),
            'epoch': tf.placeholder(tf.int32),
            'norm_feats': tf.placeholder(tf.float32),
            'pos_weight_feats': tf.placeholder(tf.float32),
        }

    model, opt = create_model(placeholders, adj, features)
    sess = tf.Session()
    
    if FLAGS.use_k_fold: # Don't use k-fold for large dataset
        raise Exception('This setting isn\'t handled!')
        
        k_adj_train, k_train_edges, k_val_edges, k_val_edges_false, test_edges, test_edges_false = load_masked_test_edges_for_kfold(dataset_str, FLAGS.k, FLAGS.split_idx)
        #k_adj_train, k_train_edges, k_val_edges, k_val_edges_false, test_edges, test_edges_false = mask_test_edges_for_kfold(adj, FLAGS.k, all_edge_idx)

        all_adj_scores = np.zeros((num_nodes, num_nodes))
        for k_idx in range(FLAGS.k):
            print (str(k_idx) + " fold")

            adj_train = k_adj_train[k_idx]
            train_edges = k_train_edges[k_idx]
            val_edges = k_val_edges[k_idx]
            val_edges_false = k_val_edges_false[k_idx]

            if FLAGS.test:
                    adj_score  = load_model(placeholders, model, opt, adj_train, test_edges, test_edges_false,
                                                        features, sess, name="k-fold-%d"%(k_idx+1))
            else:
                    adj_score, model_z = train(placeholders, model, opt, adj_train, train_edges, val_edges, val_edges_false,
                                                   test_edges, test_edges_false, features, sess, name="k-fold-%d"%(k_idx+1))
            
            all_adj_scores += adj_score

        all_adj_scores /= FLAGS.k
        roc_score, ap_score, auc_pr = get_roc_score(all_adj_scores, test_edges, test_edges_false)

    else:
        
        if FLAGS.random_split:
            adj_train, train_edges, train_edges_false = mask_train_edges(adj)
        else:
            adj_train, train_edges, train_edges_false = load_masked_train_edges(dataset_str, FLAGS.split_idx)
            adj_train = adj_train[0]
            #adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj, None)
        
        if FLAGS.test:
                adj_score  = load_model(placeholders, model, opt, adj_train, train_edges,
                                                    train_edges_false, features, sess)
        else:
                adj_score, model_z = train(placeholders, model, opt, adj_train, train_edges, train_edges_false, features, sess)

        roc_score, ap_score, auc_pr = get_roc_score(adj_score, train_edges, train_edges_false)
        all_adj_scores = adj_score
        
        if(FLAGS.semisup_train):
            semisup_acc = get_semisup_acc(model_z, y_test, test_mask)

    
    # Testing
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))
    print('Test AUC PR Curve: ' + str(auc_pr))

    if(FLAGS.semisup_train):
        print('Test Acc. :', str(semisup_acc))

    if FLAGS.log_results:
        results_log_file = save_path_disk + 'results_log_' + dataset_str + '_' + model_str + k_fold_str + FLAGS.hidden + '.log'

        #if path exists
        if not os.path.exists(os.path.dirname(results_log_file)):
            os.makedirs(os.path.dirname(results_log_file))
        
        end_time = time.time()
        time_taken = end_time - start_time

        with open(results_log_file,'a') as rlog_file:
            if FLAGS.semisup_train:
                rlog_file.write('Split: {}\nROC: {}\nAP:{}\nAUC-PR: {}\n Semisup: {}\n Time-taken (s): {}\n\n'.format(FLAGS.split_idx, str(roc_score),str(ap_score),str(auc_pr), str(semisup_acc), str(time_taken)))
            else:
                rlog_file.write('Split: {}\nROC: {}\nAP:{}\nAUC-PR: {}\nTime-taken (s): {}\n\n'.format(FLAGS.split_idx, str(roc_score),str(ap_score),str(auc_pr), str(time_taken)))

            #if FLAGS.semisup_train:
            #    rlog_file.write('Semisup-training accuracy: {}\n'.format(str(semisup_acc)))
        #print ("*"*10)
        print ("\n")
    
if __name__ == '__main__':
    main()

