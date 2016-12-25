import numpy as np
np.random.seed(1337)  # for reproducibility

from EasyNN import *

'''
    Train a simple convnet on the MNIST dataset.

    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py

    Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
    16 seconds per epoch on a GRID K520 GPU.
'''

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

sample_num=len(X_train)
test_num=len(X_test)
X_train=theano.shared(np.asarray(X_train,dtype=theano.config.floatX))
X_test=theano.shared(np.asarray(X_test,dtype=theano.config.floatX))
Y_train=theano.shared(np.asarray(Y_train,dtype=theano.config.floatX))
Y_test=theano.shared(np.asarray(Y_test,dtype=theano.config.floatX))


#setp2,define input
inputX=T.TensorType(dtype='float32',broadcastable=(False,)*4)()
inputy=T.matrix('inputy')
index=T.lscalar('input_index')
#step3,define layers
#define first layer


# model = Sequential()
#
# model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
#                         border_mode='full',
#                         input_shape=(1, img_rows, img_cols)))
cnn1=Convolution2D(nb_filter=nb_filters,nb_row=nb_conv,nb_col=nb_conv,activation='relu')
cnn1.build([None,1,img_rows,img_cols])
cnn1_out=cnn1.call(inputX)
cnn2=Convolution2D(nb_filter=nb_filters,nb_row=nb_conv,nb_col=nb_conv,activation='relu')
cnn2.build(cnn1.get_output_shape())
cnn2_out=cnn2.call(cnn1_out)
pool=MaxPooling2D(pool_size=(nb_pool,nb_pool))
pool.build(cnn2.get_output_shape())
pool_out=pool.call(cnn2_out)
pool_out1=pool_out
pool_out=Dropout(0.5).call(pool_out)
# flatten_out=T.reshape(pool_out,newshape=(pool.get_output_shape()[0],pool.get_output_shape()[1]*pool.get_output_shape()[2]*pool.get_output_shape()[3]))
flatten_out1=T.flatten(pool_out1,outdim=2)
flatten_out=T.flatten(pool_out,outdim=2)
dense1=Dense(128,activation='relu')
dense1.build([pool.get_output_shape()[0],pool.get_output_shape()[1]*pool.get_output_shape()[2]*pool.get_output_shape()[3]])
dense1_out=dense1.call(flatten_out)
dense1_out1=dense1.call(flatten_out1)
dense1_out=Dropout(0.5).call(dense1_out)
dense2=Dense(nb_classes,activation='softmax')
dense2.build(dense1.get_output_shape())
dense2_out=dense2.call(dense1_out)
dense2_out1=dense2.call(dense1_out1)

acc1=T.mean(T.eq(T.argmax(inputy, axis=-1), T.argmax(dense2_out1, axis=-1)))
acc=T.mean(T.eq(T.argmax(inputy, axis=-1), T.argmax(dense2_out, axis=-1)))
#setp4,calculate cost
cost=T.mean(categorical_crossentropy(inputy,dense2_out),axis=-1)
#setp5,define optimizer,update W,bias
sgd=SGD(lr=0.001)
sgd=Adadelta()
updates=sgd.get_updates(cnn1.trainable+cnn2.trainable+dense1.trainable+dense2.trainable,cnn1.get_constraints()+cnn2.get_constraints()+dense1.get_constraints()+dense2.get_constraints(),cost)
#step6,define theano function
train_fn=theano.function(
                        [index],
                         outputs=[cost,acc],
                         updates=updates,
                         givens={inputX:X_train[index*batch_size:(index+1)*batch_size],inputy:Y_train[index*batch_size:(index+1)*batch_size]},
                        allow_input_downcast=True
                         )
test_fn=theano.function([index],outputs=[acc1],givens={inputX:X_test[index*batch_size:(index+1)*batch_size],inputy:Y_test[index*batch_size:(index+1)*batch_size]},
                        allow_input_downcast=True)
#calculate batch_num
batch_num=(sample_num+batch_size-1)/batch_size
batch_num_test=(test_num+batch_size-1)/batch_size
print("batch num {}".format(batch_num))
#finally,began to train
epoch=20
for i in xrange(epoch):
    print(i)
    start=time.clock()
    for index in xrange(batch_num):
        loss=train_fn(index)
    print(loss)
    test_acc=0
    for index in xrange(batch_num_test):
        acc=test_fn(index)
        test_acc+=acc[0]
    print(test_acc/batch_num_test)
    end=time.clock()
    print((end-start))

# model.add(Activation('relu'))
# model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adadelta')
#
# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
# score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
