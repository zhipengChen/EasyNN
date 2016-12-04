from EasyNN import *
import theano
import theano.tensor as T
nb_classes = 10
nb_epoch = 20
#step1,init data
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
sample_num=len(X_train)
batch_size=32
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print(X_train.shape, 'train samples')
print(X_test.shape, 'test samples')
print(Y_train.shape, 'train samples label')
print(Y_test.shape, 'test samples label')
X_train=theano.shared(np.asarray(X_train,dtype=theano.config.floatX))
X_test=theano.shared(X_test)
Y_train=theano.shared(np.asarray(Y_train,dtype=theano.config.floatX))
Y_test=theano.shared(Y_test)

#setp2,define input
inputX=T.matrix('inputx')
inputy=T.matrix('inputy')
index=T.lscalar('input_index')
#step3,define layers
#define first layer
dense=Dense(400,activation='sigmoid')
dense.build(input_shape=(None,784))
dense_out=dense.call(inputX)
#define second layer
dense1=Dense(10,activation="softmax")
dense1.build((None,400))
dense1_out=dense1.call(dense_out)

#setp4,calculate cost
cost=T.mean(categorical_crossentropy(inputy,dense1_out),axis=-1)
#setp5,define optimizer,update W,bias
sgd=SGD(lr=0.001)
updates=sgd.get_updates(dense.trainable+dense1.trainable,dense.constrains+dense1.constrains,cost)
#step6,define theano function
train_fn=theano.function(
                        [index],
                         outputs=cost,
                         updates=updates,
                         givens={inputX:X_train[index*batch_size:(index+1)*batch_size],inputy:Y_train[index*batch_size:(index+1)*batch_size]},
                        allow_input_downcast=True
                         )
#calculate batch_num
batch_num=(sample_num+batch_size-1)/batch_size
print(batch_num)
#finally,began to train
epoch=10
for i in xrange(10):
    print(i)
    start=time.clock()
    for index in xrange(batch_num):
        loss=train_fn(index)
    end=time.clock()
    print((end-start))










