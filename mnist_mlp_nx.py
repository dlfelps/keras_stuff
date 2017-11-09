'''Trains a simple deep NN on the MNIST dataset.

'''

from time import time

import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, GaussianNoise
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers.core import K, Lambda

K.clear_session()

def load_data(num_classes, reps=1):    
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (_, _) = mnist.load_data()
    
    x_train = x_train.reshape(60000, 784)
    x_train = x_train.astype('float32')
    x_train /= 255
    print(x_train.shape[0], 'train samples')
    
    # find indexes of all samples with label "7"
    idx = (y_train == 7).nonzero()[0]
    
    idx = idx[0:num_classes]#keep first 1000 examples
    
    # remove all other examples
    x_train = x_train[idx,:]
    
    # create new y_train
    y_train = list(range(0,num_classes))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    
    #repeat to increase batch size
    x_train = np.tile(x_train,(reps,1))
    y_train = np.tile(y_train,(reps,1))
    
    return (x_train, y_train)


def train(num_classes=100, epochs=100, reps=1):
    (x_train, y_train) = load_data(num_classes,reps)
    model = Sequential()
    
    #model.add(GaussianNoise(0.95, input_shape=(784,)))
    model.add(Lambda(lambda x: x+K.random_normal(shape=K.shape(x), mean=0., stddev=0.10), input_shape=(784,)))#permanent noise
   
    
    #model.add(Dropout(0.95, input_shape=(784,))) #regular dropout
    model.add(Lambda(lambda x: K.dropout(x, level=0.9), input_shape=(784,)))#permanent dropout
    
    model.add(Dense(1024, activation='relu', input_shape=(784,)))
    model.add(Dense(1024, activation='relu', input_shape=(784,)))
    model.add(Dense(num_classes, activation='softmax'))
    model.load_weights("saved_noise.hdf5")
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    #add callbacks
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=100, write_graph=True, write_images=False)
    checkpoint = ModelCheckpoint("saved_noise.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=10)
    stopper = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=100, verbose=1, mode='auto')
    
    model.fit(x_train, y_train,
        batch_size=num_classes*reps,
        epochs=epochs,
        verbose=1,
        validation_data=(x_train,y_train),
        callbacks=[tensorboard, checkpoint, stopper])
    
    return model

def test(model, num_classes):
    (x_train, y_train) = load_data(num_classes)
    y_guess = model.predict(x_train, batch_size=num_classes, verbose=1)

if __name__ == '__main__':
    num_classes = 100
    epochs = 1000
    reps = 100
    model = train(num_classes, epochs, reps)
    
    
    
