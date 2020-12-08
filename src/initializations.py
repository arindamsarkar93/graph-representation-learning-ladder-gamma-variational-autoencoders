import tensorflow as tf
import numpy as np
from utils import draw_gamma

def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.get_variable(initializer=initial, name=name)

def weight_variable_gamma(input_dim, output_dim, name=""):

    shape = 1.
    rate = 0.5

    shape_mat = tf.ones([input_dim, output_dim]) * shape
    rate_mat = tf.ones([input_dim, output_dim]) * rate

    initial = draw_gamma(shape_mat, rate_mat)

    return tf.get_variable(initializer=initial, name=name)

def matrix_weight_variable_truncated_normal(dim, name="", mean=0., scale=0.01):
    initial = tf.truncated_normal((dim, dim), mean=0., stddev=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def matrix_weight_variable_normal(dim, name="", mean=0., scale=1.):
    initial = tf.random_normal((dim, dim), mean=0., stddev=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def vector_weight_variable_truncated_normal(dim, name="", mean=0., scale=0.01):
    initial = tf.truncated_normal(dim, mean=0., stddev=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)

