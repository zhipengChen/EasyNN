from __future__ import absolute_import
import theano
import theano.tensor as T

from .. import activations,  regularizers, constraints
from ..utils import initializations
from .layers import Layer
from ..utils.theano_utils import sharedX

from ..constraints import unitnorm


class Embedding(Layer):
    '''
        Turn positive integers (indexes) into denses vectors of fixed size.
        eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

        @input_dim: size of vocabulary (highest input integer + 1)
        @out_dim: size of dense representation
    '''
    input_ndim = 2

    def __init__(self, input_dim, output_dim, init='uniform', input_length=None,
                 W_regularizer=None, activity_regularizer=None, W_constraint=None,
                 mask_zero=False, weights=None, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.input_length = input_length
        self.mask_zero = mask_zero

        # self.W_constraint = constraints.get(W_constraint)
        # self.constraints = [self.W_constraint]

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)


    def build(self,input_shape):
        # self.input = T.imatrix()
        self.W = self.init((self.input_dim, self.output_dim))
        self.trainable = [self.W]
        self.regularizers = []
        # if self.W_regularizer:
        #     self.W_regularizer.set_param(self.W)
        #     self.regularizers.append(self.W_regularizer)

        # if self.activity_regularizer:
        #     self.activity_regularizer.set_layer(self)
        #     self.regularizers.append(self.activity_regularizer)
    def get_output_shape(self,input_shape):
        input_shape.append(self.output_dim)
        return input_shape
    def call(self, x,train=False):
        X = x
        out = self.W[X]
        return out



class WordContextProduct(Layer):
    '''
        This layer turns a pair of words (a pivot word + a context word,
        ie. a word from the same context, or a random, out-of-context word),
        indentified by their index in a vocabulary, into two dense reprensentations
        (word representation and context representation).

        Then it returns activation(dot(pivot_embedding, context_embedding)),
        which can be trained to encode the probability
        of finding the context word in the context of the pivot word
        (or reciprocally depending on your training procedure).

        The layer ingests integer tensors of shape:
        (nb_samples, 2)
        and outputs a float tensor of shape
        (nb_samples, 1)

        The 2nd dimension encodes (pivot, context).
        input_dim is the size of the vocabulary.

        For more context, see Mikolov et al.:
            Efficient Estimation of Word reprensentations in Vector Space
            http://arxiv.org/pdf/1301.3781v3.pdf
    '''
    input_ndim = 2

    def __init__(self, input_dim, proj_dim=128,
                 init='uniform', activation='sigmoid', weights=None, **kwargs):

        super(WordContextProduct, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)

        self.input = T.imatrix()
        # two different embeddings for pivot word and its context
        # because p(w|c) != p(c|w)
        self.W_w = self.init((input_dim, proj_dim))
        self.W_c = self.init((input_dim, proj_dim))

        self.params = [self.W_w, self.W_c]

        if weights is not None:
            self.set_weights(weights)

    @property
    def output_shape(self):
        return (self.input_shape[0], 1)

    def get_output(self, train=False):
        X = self.get_input(train)
        w = self.W_w[X[:, 0]]  # nb_samples, proj_dim
        c = self.W_c[X[:, 1]]  # nb_samples, proj_dim

        dot = T.sum(w * c, axis=1)
        dot = theano.tensor.reshape(dot, (X.shape[0], 1))
        return self.activation(dot)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "input_dim": self.input_dim,
                  "proj_dim": self.proj_dim,
                  "init": self.init.__name__,
                  "activation": self.activation.__name__}
        base_config = super(WordContextProduct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
