from layers import LinearLayer, GraphConvolution, GraphConvolutionSparse, PosInnerProductDecoder, InnerProductDecoder, DiagonalInnerProductDecoder, SparseLinearLayer
import tensorflow as tf
from utils import *
from initializations import weight_variable_glorot, weight_variable_gamma

flags = tf.app.flags
FLAGS = flags.FLAGS
SMALL = 1e-16

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass
   
class LadderGammaVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, num_classes = 0, mc_samples=1, **kwargs):
        super(LadderGammaVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.training = placeholders['is_training']
        #self.weighted_links = weighted_links

        self.hidden = [int(x) for x in FLAGS.hidden.split('_')]
        self.num_hidden_layers = len(self.hidden)

        self.prior_theta_params = [];
        self.posterior_theta_params = []
        self.z = []
        self.num_classes = num_classes
        self.S = mc_samples #No. of MC samples
        
        self.build()

    def get_regualizer_cost(self, regularizer):

        regularization = 0
        #regularization += self.last_layer.apply_regularizer(regularizer)
        
        for var in self.to_regularize:
            regularization += regularizer(var.vars['weights'])# * FLAGS.weight_decay

        return regularization
    
    def _build(self):

        print 'Build Dynamic Network....'

        self.posterior_theta_params = []
        self.d = []
        self.to_regularize = []

        # Upward Inference Pass
        for idx, hidden_layer in enumerate(self.hidden):

            #act = lambda a: tf.nn.leaky_relu(a, alpha=0.2)
            
            #This selection is questionable. May not be much of effect in reality
            if FLAGS.semisup_train:
                act = tf.nn.relu
            else:
                act = lambda x: x
            #act = tf.nn.relu
           
            """
            if idx+1 == self.num_hidden_layers:
                act = lambda x:x
            else:
                act = lambda a: tf.nn.leaky_relu(a, alpha=0.2)
            """
            
            if idx == 0:
                gc = GraphConvolutionSparse(input_dim=self.input_dim,
                                            output_dim=hidden_layer,
                                            adj=self.adj,
                                            features_nonzero=self.features_nonzero,
                                            act=act,#tf.nn.relu
                                            name = "conv_weight_"+str(idx),
                                            dropout=self.dropout,
                                            logging=self.logging)
                self.to_regularize.append(gc)
                d = gc(self.inputs)
            
                if(False and FLAGS.features==1 and FLAGS.reconstruct_x==1):
                    #feature transform
                    x_h = SparseLinearLayer(input_dim=self.input_dim,
                                        output_dim=hidden_layer,
                                        dropout=self.dropout,
                                        features_nonzero=self.features_nonzero,
                                        reuse_name = "conv_weight_"+str(idx),
                                        reuse = True)(self.inputs)
                    x_h = tf.nn.relu(x_h)

            else:
                gc = GraphConvolution(input_dim=self.hidden[idx-1],
                                     output_dim=hidden_layer, #self.num_classes,
                                     adj=self.adj,
                                     act = act,#tf.nn.relu,
                                     dropout=self.dropout,
                                     name = "conv_weight_"+str(idx),
                                     logging=self.logging)

                self.to_regularize.append(gc)
                d = gc(d)

                if(False and FLAGS.features==1 and FLAGS.reconstruct_x==1):
                    #feature transform
                    x_h = LinearLayer(input_dim=self.hidden[idx-1],
                                        output_dim=hidden_layer,
                                        dropout=self.dropout,
                                        reuse_name = "conv_weight_"+str(idx),
                                        reuse = True)(x_h)
                                    
                    x_h = tf.nn.relu(x_h)
            d = tf.nn.l2_normalize(d, dim=1)
            self.d.append(d) 

            # get gamma params
            shape_layer = LinearLayer(input_dim=hidden_layer, output_dim=hidden_layer)
            self.to_regularize.append(shape_layer)
            shape = shape_layer(d)
            shape = tf.nn.softplus(shape)

            #print shape.get_shape() 
            rate_layer = LinearLayer(input_dim=hidden_layer, output_dim=hidden_layer)
            self.to_regularize.append(rate_layer)
            rate = rate_layer(d)
            rate = tf.nn.softplus(rate)
            self.posterior_theta_params.append([shape, rate])
            #print 'Shape:' + str(shape.get_shape())
            

        # Downward Inference pass
        # Careful: The posterior order is reverse of prior (and theta samples).
        # We will invert array after downward pass is complete
        # Note that in formulation here, Weibull has shape,scale parametrization while gamma has shape,rate parametrization
        prior_shape_const = 10e-5
        prior_rate_const = 10e-3
        prior_rate_const2 = 10e-2

        #Merged!
        #x_h = d

        self.reconstructions_list = []
        self.posterior_theta_params_list = []
        self.prior_theta_params_list = []
        self.phi = []
        self.reg_phi = 0.
        posterior_theta_params_cp = self.posterior_theta_params
        
        ###########################################################################
        #Take multiple MC samples
        for s in range(self.S):
            # Refresh
            theta = []
            self.prior_theta_params = []
            self.posterior_theta_params = posterior_theta_params_cp
            
            # Processing top layer first
            theta_sample = draw_weibull(self.posterior_theta_params[-1][0], self.posterior_theta_params[-1][1])
 
            theta.append(theta_sample)

            prior_1 = np.ones((self.n_samples, self.hidden[-1])) * prior_shape_const
            prior_2 = np.ones((self.n_samples, self.hidden[-1])) * prior_rate_const
            
            prior_1 = tf.constant(prior_1, tf.float32) 
            prior_2 = tf.constant(prior_2, tf.float32) 

            self.prior_theta_params.append([prior_1, prior_2]) # arbritrary

            parent_gamma_theta = draw_gamma(prior_1, prior_2)

            self.shape_d = []
            
            # Downward Inference Pass
            for idx in range(self.num_hidden_layers-2, -1, -1):
                
                with tf.variable_scope("phi_"+str(idx), reuse = tf.AUTO_REUSE): 
                    phi = weight_variable_gamma(self.hidden[idx+1], self.hidden[idx])#, name='phi' + '_' + str(idx))
                    #phi = weight_variable_glorot(self.hidden[idx+1], self.hidden[idx])#, name='phi' + '_' + str(idx))
                    phi = tf.nn.softmax(phi, axis=0)
                    
                    #self.reg_phi += tf.nn.l2_loss(phi)
                    #Will be messed up if you take mc-samples!!!
                    self.phi.append(phi)
                
                #print len(self.hidden)
                top_theta = theta[self.num_hidden_layers - idx - 2]
                
                self.shape_d = tf.matmul(top_theta, phi)
                #self.shape_d = tf.nn.sigmoid(self.shape_d)
                
                self.posterior_theta_params[idx][0] += self.shape_d
               
                prior_param_shape = tf.matmul(parent_gamma_theta, phi)

                prior_1 = np.ones((self.n_samples, self.hidden[idx])) * prior_shape_const
                prior_1 = tf.constant(prior_1, tf.float32) 
                #prior_param_shape = prior_1
                
                prior_2 = np.ones((self.n_samples, self.hidden[idx])) * prior_rate_const2
                prior_2 = tf.constant(prior_2, tf.float32)
                
                self.prior_theta_params.append([prior_param_shape, prior_2])
                
                parent_gamma_theta = draw_gamma(prior_param_shape, prior_2)

                theta_sample = draw_weibull(self.posterior_theta_params[idx][0], self.posterior_theta_params[idx][1])
                
                theta.append(theta_sample)
               
                #False deactivates :)
                if(False and FLAGS.features==1 and FLAGS.reconstruct_x==1):
                    #feature decoder (starting from L-1th layer)
                    if idx==0:
                        #last layer remains real
                        act_x = lambda x: x
                    else:
                        act_x = tf.nn.relu
                    x_h = LinearLayer(input_dim=self.hidden[idx+1], output_dim = self.hidden[idx], reuse_name = 'conv_weight_'+str(idx+1), reuse = True, transpose = True)(x_h)
                    x_h = act_x(x_h)

            # reverse
            self.theta = theta[::-1] 
            self.theta_concat = tf.concat(self.theta, axis=1)
            #print(self.theta_concat.get_shape().as_list())

            self.prior_theta_params = self.prior_theta_params[::-1]

            transformed_theta = self.theta[0] #LinearLayer(input_dim = self.hidden[i],output_dim = self.hidden[i])(self.theta[i]) 
            weight_layer = InnerProductDecoder(input_dim=self.hidden[0],
                                                   act=lambda x: x,#tf.nn.sigmoid,
                                                   normalize = FLAGS.cosine_norm,
                                                   logging=self.logging)
            self.last_layer = weight_layer

            #self.reconstructions_list.append(weight_layer(transformed_theta))
            
            #epsilon = tf.constant(10e-10)
            #self.poisson_rate = tf.clip_by_value(self.reconstructions_list[0], epsilon, 10e10)
            #self.rate = 1 - tf.exp(-self.poisson_rate)
            
            self.poisson_rate = weight_layer(transformed_theta) #self.reconstructions_list[s]
            
            if FLAGS.data_type == 'binary':
                self.clipped_logit = tf.clip_by_value(self.poisson_rate,0.0,1.0)  #tf.log(self.poisson_rate) #tf.log(self.rate)
            else:
                self.clipped_logit = self.poisson_rate

            self.reconstructions = self.clipped_logit
            self.reconstructions_list.append(self.reconstructions)
            self.posterior_theta_params_list.append(self.posterior_theta_params)
            self.prior_theta_params_list.append(self.prior_theta_params)
        ###############################################################
        
        if(FLAGS.features==1 and FLAGS.reconstruct_x==1):
            #last transform for Decoder
            x_h = LinearLayer(input_dim = self.hidden[0], output_dim = self.input_dim, reuse_name = 'conv_weight_'+str(0), reuse = True, transpose = True)(self.theta[0])
            #No non-linearity for last layer?
            self.x_recon = tf.reshape(x_h, [-1])

        if(FLAGS.semisup_train == 1):
            #can use final or prefinal layer or last d
            """
            self.z =  GraphConvolution(input_dim=self.hidden[-1],
                                     output_dim=self.num_classes,
                                     adj=self.adj,
                                     act = tf.nn.relu, #tf.nn.relu,
                                     dropout=self.dropout,
                                     name = "conv_weight_classify_"+str(idx),
                                     logging=self.logging)(d)
            """


            def weibull_mean(params):
                k = params[0]
                l = params[1]

                return l * tf.exp(tf.lgamma(1 + 1/k))

            classification_layer = LinearLayer(input_dim = self.hidden[0], output_dim = self.num_classes, name = 'semisup_weight',dropout=0.)
            self.z = classification_layer(self.theta[0])

        self.lambda_mat = self.last_layer.get_weight_matrix()

