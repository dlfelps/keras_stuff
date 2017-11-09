'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
import numpy as np

from time import time

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Lambda, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras import backend as K

K.clear_session()

batch_size = 128
# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)


def load_data(num_classes, reps=1):    
   
   
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (_, _) = mnist.load_data()
    
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    
    # find indexes of all samples with label "7"
    idx = (y_train == 7).nonzero()[0]
    
    idx = idx[0:num_classes]#keep first 100 examples
    
    # remove all other examples
    x_train = x_train[idx,:,:]
    
    x_train = x_train.astype('float32')
    x_train /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    
    # create new y_train
    y_train = list(range(0,num_classes))
    y_train = keras.utils.to_categorical(y_train, num_classes)

    #repeat to increase batch size
    x_train = np.tile(x_train,(reps,1,1,1))
    y_train = np.tile(y_train,(reps,1))
    
    return (x_train, y_train)

def train(num_classes=100, epochs=100, reps=1):
    (x_train, y_train) = load_data(num_classes,reps)

    model = Sequential()
    model.add(Lambda(lambda x: K.dropout(x, level=0.9), input_shape=input_shape))#permanent dropout

    model.add(Conv2D(32, kernel_size=(3, 3), kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(64, (3, 3),  kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    
    model.add(Dense(128,  kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(num_classes, activation='softmax'))
    
#    model.load_weights("saved_bn.hdf5")
    model.summary()

    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    #add callbacks
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=10, write_graph=True, write_images=True)
    checkpoint = ModelCheckpoint("saved_bn.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    stopper = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=10, verbose=1, mode='auto')
    
    model.fit(x_train, y_train,
        batch_size=100,
        epochs=epochs,
        verbose=1,
        shuffle=True,
        validation_data=(x_train,y_train),
        callbacks=[tensorboard, checkpoint, stopper])
    
    return model

if __name__ == '__main__':
    num_classes = 100
    epochs = 1000
    reps = 100
    model = train(num_classes, epochs, reps)

