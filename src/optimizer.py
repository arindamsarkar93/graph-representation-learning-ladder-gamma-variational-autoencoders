import tensorflow as tf
from utils import *

flags = tf.app.flags
FLAGS = flags.FLAGS

SMALL = 1e-16

class Optimizer(object):
    def __init__(self, labels, model, epoch, num_nodes, features, pos_weight, norm, weighted_ce, edges_for_loss, norm_feats, pos_weight_feats, node_labels = None, node_labels_mask = None, start_semisup=1.):

        labels_sub = labels
        pos_weight_mod = pos_weight
        epoch = tf.cast(epoch, tf.float32)

        if weighted_ce == 0:
            # Loss not weighted
            norm = 1
            pos_weight = 1

        else:
            pos_weight_mod = pos_weight * FLAGS.bias_weight_1
            
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        preds_sub = model.reconstructions
        #neg_ll = tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight_mod)
        
        # S MC samples:
        self.nll = tf.constant(0.0)
        if(FLAGS.link_prediction):
            for s in range(model.S):
                preds_sub = model.reconstructions_list[s]
                
                if FLAGS.data_type == "count":
                    neg_ll = tf.nn.log_poisson_loss(labels_sub, tf.log(preds_sub)) 
                else:
                    neg_ll = self.binary_weighted_crossentropy(preds_sub, labels_sub, pos_weight)
                
                
                neg_ll = neg_ll * edges_for_loss
                neg_ll = norm * tf.reduce_mean(neg_ll)

                self.nll += neg_ll

            self.nll = self.nll / model.S 
        self.check = model.reconstructions_list

        #else:
        #self.nll = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        
        #Regularization Loss
        self.regularization = model.reg_phi + model.get_regualizer_cost(tf.nn.l2_loss)
       
       #self.regularization = tf.reduce_mean([tf.nn.l2_loss(v) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = ".*semisup_weight_vars/weights")])# scope = ".*weights")])
        # X-Reconstruction Loss
        self.x_loss = tf.constant(0.0)
        if(FLAGS.features==1 and FLAGS.reconstruct_x==1):
            #X (features) reconstruction loss
            x_recon = model.x_recon
            #pos_weight = 1
            #0-1 features. weighing required
            self.x_loss = tf.nn.weighted_cross_entropy_with_logits(logits=x_recon, targets=features, pos_weight=pos_weight_feats)
            self.x_loss = tf.reduce_mean(self.x_loss) * norm_feats

        # Classification loss in case of semisupervised training
        self.semisup_loss = tf.constant(0.0)
        self.semisup_acc = tf.constant(0.0)
        if(FLAGS.semisup_train):

            #Only use for finetuning

            preds = model.z
            mask = node_labels_mask

            #preds_softmax = tf.nn.softmax(preds, axis = 1)
            #self.entropy = tf.reduce_mean(tf.reduce_sum(-preds_softmax * tf.log(preds_softmax + SMALL2), axis=1), axis=0)

            self.semisup_acc = masked_accuracy(preds, node_labels,  mask)
            
            loss = tf.nn.softmax_cross_entropy_with_logits(logits = preds, labels = node_labels)
            mask = tf.cast(mask, dtype = tf.float32)
            mask = mask/tf.reduce_mean(mask)
            self.semisup_loss = tf.reduce_mean(loss * mask)

            #start semisupervised training after a while
            self.semisup_loss = self.semisup_loss * start_semisup


        # KL-divergence loss
        self.kl_term = 0
        for s in range(model.S):
            for idx in range(len(model.hidden)):
                k_w = model.posterior_theta_params_list[s][idx][0]
                l_w = model.posterior_theta_params_list[s][idx][1]
                alpha_g = model.prior_theta_params_list[s][idx][0]
                beta_g = model.prior_theta_params_list[s][idx][1]
                self.kl_term += kl_weibull_gamma(k_w, l_w, alpha_g, beta_g) / num_nodes
        #Average
        self.kl_term = self.kl_term/model.S


        self.wu_beta = epoch/FLAGS.epochs;
        #self.wu_beta = self.wu_beta
        if FLAGS.use_kl_warmup == 0:
            self.wu_beta = 1

        if FLAGS.use_x_warmup == 0:
            self.wu_x = 1
        else:
            self.wu_x = self.wu_beta

        self.wu_semisup_loss = 1. #* epoch/FLAGS.epochs


        self.ae_loss = self.nll + self.regularization * FLAGS.weight_decay
        self.cost = 1. * self.nll + 1. * self.wu_beta * self.kl_term +  self.wu_x * self.x_loss + self.wu_semisup_loss * self.semisup_loss  + FLAGS.weight_decay * self.regularization # + self.entropy * 1.

        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        #gradient clipping
        #self.grad_vars = tf.clip_by_value(self.grads_vars, -5,5)
        self.clipped_grads_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else 0, var)
                for grad, var in self.grads_vars]

        self.opt_op = self.optimizer.apply_gradients(self.clipped_grads_vars)

        
        with tf.variable_scope("inputs"):
            tf.summary.histogram('label', labels)

        with tf.variable_scope("predictions"):
            tf.summary.histogram('outputs', tf.nn.sigmoid(preds_sub))

        with tf.variable_scope("loss"):
            tf.summary.scalar('cost', self.cost)
            tf.summary.scalar('nll', self.nll)
            tf.summary.scalar('kl_divergence', self.kl_term)
            tf.summary.scalar('regularization', self.regularization)
        if(FLAGS.features==1 and FLAGS.reconstruct_x==1):
            tf.summary.scalar('x_recon_loss', self.x_loss)
       
        # Add histograms for gradients.
        with tf.variable_scope("gradients"):
            for grad, var in self.grads_vars:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
    
        #self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.nn.sigmoid(preds_sub), 0.5), tf.int32),
        #                                   tf.cast(labels_sub, tf.int32))
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


    def binary_weighted_crossentropy(self, preds, labels, pos_weight):
        """
        Expects probabilities preds
        pos weight: scaling factor for 1-labels for use in unbalanced datasets with lots of zeros(?)
        See this thread for more: https://github.com/tensorflow/tensorflow/issues/2462
        """
        SMALL_VAL = 10e-8
        epsilon = tf.constant(SMALL_VAL)
        preds = tf.clip_by_value(preds, epsilon, 1-epsilon)
        #preds = tf.log(preds)
        
        loss = pos_weight * labels * -tf.log(preds) + (1 - labels) * -tf.log(1 - preds)
        
        neg_ll = loss
        return neg_ll

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
 
    return tf.reduce_mean(accuracy_all)
