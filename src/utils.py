from __future__ import division
import tensorflow as tf
import numpy as np
import sys
from tensorflow.contrib.distributions import Gamma

SMALL = 1e-16
SMALL2 = 10e-8
EULER_GAMMA = 0.5772156649015329

"""
def monte_carlo_sample_for_weighted_links(z_mean, z_log_std, pi_logit, W, temp, S, sigmoid_fn, c=0, e=None):

    shape = list(np.shape(z_mean))
    shape.insert(0, S)
    
    # mu + standard_samples * stand_deviation
    z_real = z_mean + np.multiply(np.random.normal(0, 1, shape), np.exp(z_log_std))
    
    # Concrete instead of Bernoulli => equivalent to reparametrize_discrete in tensorflow
    uniform = np.random.uniform(1e-4, 1. - 1e-4, shape)
    logistic = np.log(uniform) - np.log(1 - uniform)
    y_sample = (pi_logit + logistic) / temp
        
    z_discrete = sigmoid_fn(y_sample)
    z_discrete = np.round(z_discrete)
    
    z_activated = np.sum(z_discrete) / (shape[0] * shape[1])
    
    emb = np.multiply(z_real, z_discrete)
    emb_t = np.transpose(emb, (0, 2, 1))

    adj_rec = np.matmul(np.matmul(emb, W), emb_t)
    #adj_rec = np.matmul(emb + np.matmul(emb, W), emb_t)
    adj_rec = np.mean(adj_rec, axis=0)
    
    adj_rec += np.tile(c, np.shape(adj_rec))

    if e is not None:
        e_bias = np.tile(e, (1,np.shape(adj_rec)[0]))
        e_bias = e_bias + np.transpose(e_bias)
        adj_rec += e_bias
        
    return adj_rec, z_activated
"""

def logit(x):
    return tf.log(x + SMALL2) - tf.log(1. - x + SMALL2)

def log_density_logistic(logalphas, y_sample, temp):
    """
    log-density of the Logistic distribution, from 
    Maddison et. al. (2017) (right after equation 26)
    Input logalpha is a logit (alpha is a probability ratio)
    """
    exp_term = logalphas + y_sample * -temp
    log_prob = exp_term + np.log(temp) - 2. * tf.nn.softplus(exp_term)
    return log_prob

def Beta_fn(a, b):
    beta_ab = tf.exp(tf.lgamma(a) + tf.lgamma(b) - tf.lgamma(a + b))
    return beta_ab

def log_pdf_bernoulli(x,p):
    return x * tf.log(p + SMALL) + (1-x) * tf.log(1-p + SMALL)


def draw_weibull(k, l):
    # x ~ Weibull(k, l)
    #print k.shape
    uniform = tf.random_uniform(tf.shape(k), 1e-4, 1. - 1e-4)
    x = l * tf.pow(-tf.log(1-uniform), 1/k)
    return x

def kl_weibull_gamma(k_w, l_w, alpha_g, beta_g):
    # KL(Weibull(k_w, l_w)||Gamma(alpha_g, beta_g))
    # See eqn from paper.

    #print k_w.get_shape()
    #print l_w.get_shape()
    #print alpha_g.shape
    #sys.exit()
    
    #typecasting as log needed one of the floats
    k_w = tf.cast(k_w, tf.float32)
    l_w = tf.cast(l_w, tf.float32)
    alpha_g = tf.cast(alpha_g, tf.float32)
    beta_g = tf.cast(beta_g, tf.float32)

    kl = -alpha_g * tf.log(l_w + SMALL2) + (EULER_GAMMA * alpha_g) / (k_w+SMALL2) + tf.log(k_w + SMALL2) + beta_g * l_w * tf.exp(tf.lgamma(1 + (1 / (k_w+SMALL2) ))) - \
    EULER_GAMMA - 1 - alpha_g * tf.log(beta_g + SMALL2) + tf.lgamma(alpha_g+SMALL2)

    kl = tf.reduce_mean(tf.reduce_sum(kl, 1))
    #kl = tf.minimum(10e8, kl)
    #kl = tf.clip_by_value(kl, 0.0, 100.0)
   
    #Desperate times, desperate measures
    kl = tf.clip_by_value(kl, 0.0, 10e5)

    return kl

def draw_gamma (shape, rate):

    #rate = 1 / scale
    dist = Gamma(concentration=shape, rate=rate)
    samples = dist.sample()
    return samples


def test_kl():

    from math import gamma
    
    k_w = 2
    l_w = 2

    alpha_g = 2
    beta_g = 2

    kl = -alpha_g * np.log(l_w) + (EULER_GAMMA * alpha_g) / k_w + np.log(k_w) + beta_g * l_w * gamma(1 + (1/k_w)) - \
         EULER_GAMMA - 1 - alpha_g * np.log(beta_g) + np.log(gamma(alpha_g))

    print kl
    #sys.exit()
    
