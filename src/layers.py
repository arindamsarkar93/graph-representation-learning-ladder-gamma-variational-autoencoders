from initializations import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

class LinearLayer(Layer):
    def __init__(self, input_dim, output_dim, dropout=0., reuse_name='', reuse=False, transpose = False, **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        # reuse : for weight reuse -- tied weights
        # transpose : if reuse => for decoder part
        
        if(reuse):
            #reuse conv weights 
            with tf.variable_scope(reuse_name + '_vars', reuse = True):
                self.vars['weights'] = tf.get_variable('weights')

                if(transpose):
                    self.vars['weights'] = tf.transpose(self.vars['weights'])
                print(self.vars['weights'].name)
        
        else:
            with tf.variable_scope(self.name + '_vars'):
                self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        
        with tf.variable_scope(self.name + '_vars'):
            self.vars['bias'] = tf.Variable(tf.zeros((output_dim)), name="bias")
        self.dropout = dropout

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        output = tf.matmul(x, self.vars['weights']) + self.vars['bias']
        return output
        
class SparseLinearLayer(Layer):
    def __init__(self, input_dim, output_dim, features_nonzero, dropout=0., reuse_name='', reuse=False, transpose = False, **kwargs):
        super(SparseLinearLayer, self).__init__(**kwargs)

        # reuse : for weight reuse -- tied weights
        # transpose : if reuse => for decoder part

        if(reuse):
            #reuse conv weights 
            with tf.variable_scope(reuse_name + '_vars', reuse = True):
                self.vars['weights'] = tf.get_variable('weights')

                if(transpose):
                    self.vars['weights'] = tf.transpose(self.vars['weights'])
                print(self.vars['weights'].name)
        
        else:
            with tf.variable_scope(self.name + '_vars'):
                self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        
        with tf.variable_scope(self.name + '_vars'):
            self.vars['bias'] = tf.Variable(tf.zeros((output_dim)), name="bias")
        self.dropout = dropout
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        output = tf.sparse_tensor_dense_matmul(x, self.vars['weights']) + self.vars['bias']
        return output

class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
            #self.vars['bias'] = tf.Variable(tf.zeros((output_dim)), name="bias")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights']) #+ self.vars['bias']
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs

    def apply_regularizer(self, regularizer):
        return 0
        #return regularizer(self.vars['weights'])

    
class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
            #self.vars['bias'] = tf.Variable(tf.zeros((output_dim)), name="bias")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights']) #+ self.vars['bias']
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs

    def apply_regularizer(self, regularizer):
        return 0
        #return regularizer(self.vars['weights'])


class WeightedInnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(WeightedInnerProductDecoder, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_weight'):
            self.vars['weights'] = matrix_weight_variable_truncated_normal(input_dim, name="matrix_weight")
        self.dropout = dropout
        self.act = act

    def get_weight_matrix(self):
        W = (self.vars['weights'] + tf.transpose(self.vars['weights'])) * 1/2
        return W
    
    def _call(self, inputs):

        W = (self.vars['weights'] + tf.transpose(self.vars['weights'])) * 1/2
        
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        #inputs = inputs + tf.matmul(inputs, W)
        inputs = tf.matmul(inputs, W)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs

    def apply_regularizer(self, regularizer):
        return regularizer(self.vars['weights'])

class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., normalize = False, act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.input_dim = input_dim
        self.normalize = normalize

    def _call(self, inputs):
        if(self.normalize):
            inputs = tf.nn.l2_normalize(inputs, dim=1)
        
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs
    
    def get_weight_matrix(self):
        W = tf.eye(self.input_dim)
        return W

    def apply_regularizer(self, regularizer):
        return tf.constant(0.0)

class WeightedInnerProductDecoder2(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(PosInnerProductDecoder, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_weight'):
            self.vars['weights'] = matrix_weight_variable_normal(input_dim, scale=FLAGS.lambda_mat_scale, name="matrix_weight")

        self.dropout = dropout
        self.act = act

    def get_weight_matrix(self):
        W = self.vars['weights']
        W = (W + tf.transpose(W)) * 1/2
        #W = tf.nn.sigmoid(W)
        #W = tf.nn.softmax(W);
        return W
    
    def _call(self, inputs):

        W = self.get_weight_matrix()
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        #inputs = inputs + tf.matmul(inputs, W)
        inputs = tf.matmul(inputs, W)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
	
        return outputs

    def apply_regularizer(self, regularizer):
        return regularizer(self.vars['weights'])

class PosInnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(PosInnerProductDecoder, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_weight'):
            self.vars['weights'] = matrix_weight_variable_normal(input_dim, scale=FLAGS.lambda_mat_scale, name="matrix_weight")

        self.dropout = dropout
        self.act = act

    def get_weight_matrix(self):
        W = self.vars['weights']
        W = (W + tf.transpose(W)) * 1/2
        W = tf.nn.sigmoid(W)
        #W = tf.nn.softmax(W);
        return W
    
    def _call(self, inputs):

        W = self.get_weight_matrix()
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        #inputs = inputs + tf.matmul(inputs, W)
        inputs = tf.matmul(inputs, W)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
	
        return outputs

    def apply_regularizer(self, regularizer):
        return regularizer(self.vars['weights'])

class DiagonalInnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., normalize = False, act=tf.nn.sigmoid, **kwargs):
        super(DiagonalInnerProductDecoder, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_weight'):
            self.vars['weights'] = vector_weight_variable_truncated_normal((1, input_dim), name="matrix_weight", scale=0.1)

        self.dropout = dropout
        self.act = act
        self.normalize = normalize
    
    def _call(self, inputs):

        if(self.normalize):
            inputs = tf.nn.l2_normalize(inputs, dim=1)
        
        W = self.get_weight_matrix()#self.vars['weights'];#tf.nn.sigmoid(self.vars['weights'])
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        inputs = inputs * W
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
	
        return outputs

    def apply_regularizer(self, regularizer):
        return regularizer(self.vars['weights'])

    
    def get_weight_matrix(self):
        W = self.vars['weights']
        #W = (W + tf.transpose(W)) * 1/2
        W = tf.nn.sigmoid(W)
        #W = tf.nn.softmax(W);
        return W

class batch_norm(object):
     
     #def __init__(self, epsilon=1e-5, momentum = 0.99, name="batch_norm"):
     def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
          with tf.variable_scope(name):
               self.epsilon  = epsilon
               self.momentum = momentum
               self.name = name

     def __call__(self, x, phase):
          return tf.contrib.layers.batch_norm(x,
                                              decay=self.momentum, 
                                              epsilon=self.epsilon,
                                              scale=True,
                                              center=True, 
                                              is_training=phase,
                                              scope=self.name)
