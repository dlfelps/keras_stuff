'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from time import time

import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, GaussianNoise
from keras.callbacks import TensorBoard
from keras.layers.core import K, Lambda

K.clear_session()


batch_size = 6265
epochs = 1000

# the data, shuffled and split between train and test sets
(x_train, y_train), (_, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_train = x_train.astype('float32')
x_train /= 255
print(x_train.shape[0], 'train samples')

# find indexes of all samples with label "7"
idx = (y_train == 7).nonzero()[0]
idx = idx[0:100]#keep first 1000 examples

# remove all other examples
x_train = x_train[idx,:]


# create new y binary y_train
len_z = int(idx.shape[0]/2)
z = np.zeros(len_z, dtype="float32")
o = np.ones(idx.shape[0]-len_z, dtype="float32")
y_train = np.append(z,o)


model = Sequential()
#model.add(Dropout(0.95, input_shape=(784,))) #regular dropout
model.add(Lambda(lambda x: K.dropout(x, level=0.9), input_shape=(784,)))#permanent dropout

#model.add(GaussianNoise(0.01, input_shape=(784,)))
model.add(Dense(1024, activation='relu', input_shape=(784,)))
model.add(Dense(1024, activation='relu', input_shape=(784,)))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=10,  
          write_graph=True, write_images=False)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_train,y_train),
                    callbacks=[tensorboard])
