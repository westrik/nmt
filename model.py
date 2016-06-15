# -*- coding: utf-8 -*-
'''
RNN encoder-decoder architecture
for statistical machine translation
'''

from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import keras


class RNNEncDecAttnModel:
    def __init__(self, cp):
	print("Building network ...")
       
	# First, we build the network, starting with an input layer
	# Recurrent layers expect input of shape
	# (batch size, SEQ_LENGTH, num_features)




    def train(self):
        pass

'''
how to implement maxout layer:

l1a = lasagne.layers.DenseLayer(l_in, nonlinearity=None, num_units=512, ...)
l1 = lasagne.layers.FeaturePoolLayer(l1a, ds=2)

not sure how many units to use or what to connect this to
'''
