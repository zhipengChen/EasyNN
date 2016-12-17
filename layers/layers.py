import theano
import theano.tensor as T
from ..utils import initializations
from ..utils import *
from ..activations import activations
from .. import constraints
class Layer(object):
    def __init__(self,**kwargs):
        print('init layer')
        # if self.trainable == None:
        self.trainable=[]
        self.constraints=[constraints.get('Constraint')]
    def get_weights(self):
        return
    def set_weights(self):
        return
    def build(self,input_shape):
        print('build')
        raise NotImplementedError
    def get_constraints(self):
        return self.constraints*len(self.trainable)
    def call(self,x):
        raise NotImplementedError
    def get_output_shape(self):
        raise NotImplementedError
class Dense(Layer):
    def __init__(self,output_shape,activation="",init='glorot_uniform',**kwargs):
        super(Dense, self).__init__(**kwargs)
        print('test')
        self.init=initializations.get(init)
        self.output_shape=output_shape
        self.activation=activations.get(activation)
    def build(self,input_shape):
        self.W=self.init((input_shape[-1],self.output_shape))
        self.b=shared_zeros((self.output_shape,))
        self.trainable=[self.W,self.b]
    def call(self,x):
        return self.activation(T.dot(x,self.W)+self.b)
    def get_output_shape(self,input_shape):
        return (input_shape[0],self.output_shape)
class Activation(Layer):
    def __init__(self,activation="sigmoid",**kwargs):
        self.activation=activations.get(activation)
        super(Activation,self).__init__(**kwargs)
    def call(self,x):
        return self.activation(x)