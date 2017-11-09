#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 20:20:58 2017

@author: dlfelps
"""

import keras
import numpy as np

from keras import activations
from keras.models import Model, load_model, Sequential
from keras.layers import Input

from vis.visualization import visualize_activation
from vis.utils import utils
from vis.input_modifiers import Jitter



from matplotlib import pyplot as plt

from mnist_cnnx_bn import load_data

def merge_models(model1, model2):
    '''
    This function combines two models such that:
    input -> model1 -> model2 -> output 
    or equivalently, output = model2(model1(input))
    
    The output of model1.output_shape must be the same as model2.input_shape.
    
    The models can be written using the functional or Sequential API.
    The returned model is the same type as model1.
    '''
    
    #First convert both models to Sequential type
    func_flag = False
    if type(model1) is not Sequential:
        model1 = Sequential(layers=model1.layers)
        func_flag = True

    if type(model2) is not Sequential:
        model2 = Sequential(layers=model2.layers)
        
    #at this point both models should be Sequential
    #now pop n-1 layers out of model2
    temp = []

    for i in range(0,len(model2.layers)-1):
        temp.append(model2.layers.pop())#don't pop the first layer (i.e. input tensor)

    #now add those popped layers to model1
    for i in range(0,len(temp)):
        new_layer = temp.pop()
        model1.add(new_layer)
        
    new_model = utils.apply_modifications(model1)
   
    return new_model
    

model = keras.models.load_model("saved_bn.hdf5")
layer_idx = len(model.layers)-1#pick last layer

#change activation function on last layer from softmax to linear
model.layers[layer_idx].activation = activations.linear

#turn off dropout layer (just pass through input)
model.layers[0].function = lambda x: x
model = utils.apply_modifications(model)

#extract functional portion of model
model = model.model

#attach generator portion of variational autoencoder
generator = load_model('generator_cnn.hdf5')
combined = merge_models(generator, model)
layer_idx = len(combined.layers)-1#pick last layer

#generate seed input

filter_idx = 10
x_train,_ = load_data(100,1)
encoder = keras.models.load_model('encoder_cnn.hdf5')
seed_input = encoder.predict(x_train[filter_idx,:,:,:].reshape(1,28,28,1))

code = visualize_activation(combined, layer_idx, filter_indices=filter_idx, input_range=(-3.,3.), 
                           seed_input=seed_input, lp_norm_weight=0, tv_weight=0, 
                           max_iter=100, verbose=True, input_modifiers=[], callbacks=[])
img = generator.predict(seed_input.reshape((1,5)))

img = img.squeeze()
plt.imshow(img,cmap='gray')
plt.show()
plt.imshow(x_train[filter_idx,:,:,0], cmap='gray')



