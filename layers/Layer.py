import theano
import theano.tensor as T
from ..utils import initializations
from ..utils import *
from ..activations import activations
from .. import constrains
class Layer(object):
    def __init__(self,**kwargs):
        print('init layer')
        self.trainable=[]
        self.constrains=[constrains.get('Constraint')]
    def get_weights(self):
        return
    def set_weights(self):
        return
    def build(self):
        print('build')
    def call(self):
        return
    def get_output_shape(self):
        return
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