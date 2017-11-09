'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
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

num_classes = 100
batch_size = num_classes
epochs = 10000

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

model = Sequential()
#model.add(Dropout(0.95, input_shape=(784,))) #regular dropout
model.add(Lambda(lambda x: K.dropout(x, level=0.9), input_shape=(784,)))#permanent dropout

#model.add(GaussianNoise(0.95, input_shape=(784,)))
model.add(Dense(1024, activation='relu', input_shape=(784,)))
model.add(Dense(1024, activation='relu', input_shape=(784,)))
model.add(Dense(num_classes, activation='softmax'))
model.load_weights("saved.hdf5")
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#add callbacks
tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=100, write_graph=True, write_images=False)
checkpoint = ModelCheckpoint("saved.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=100)
stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=500, verbose=1, mode='auto')


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_train,y_train),
                    callbacks=[tensorboard, checkpoint, stopper])
