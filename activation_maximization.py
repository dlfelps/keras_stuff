#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 20:20:58 2017

@author: dlfelps
"""

import keras
import numpy as np

from keras import activations

from vis.visualization import visualize_activation
from vis.utils import utils
from vis.input_modifiers import Jitter
from vis.grad_modifiers import invert
from vis.callbacks import GifGenerator


from matplotlib import pyplot as plt

from mnist_cnnx_bn import load_data


model = keras.models.load_model("saved_bn.hdf5")
layer_idx = len(model.layers)-1#pick last layer

#change activation function on last layer from softmax to linear
model.layers[layer_idx].activation = activations.linear

#turn off dropout layer (just pass through input)
model.layers[0].function = lambda x: x
model = utils.apply_modifications(model)

#generate seed input

filter_idx = 10
x_train,_ = load_data(100,1)
seed_input = x_train.mean(axis=0)
seed_input = x_train[filter_idx,:,:,:]

img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0.,1.), 
                           seed_input=seed_input, lp_norm_weight=0, tv_weight=0, 
                           max_iter=100, verbose=True, input_modifiers=[], callbacks=[])
plt.imshow(img[:,:,0],cmap='gray')
plt.show()
plt.imshow(x_train[filter_idx,:,:,0], cmap='gray')



