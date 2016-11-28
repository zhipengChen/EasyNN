import theano
import theano.tensor as T
class Layer():
    def __init__(self):
        print('init layer')
        self.trainable=[]

    def get_weights(self):
        return
    def set_weights(self):
        return
    def build(self):
        print('build')
    def call(self):
        return
class Dense(Layer):
    def __init__(self):
        print('test')