# -*- coding: utf-8 -*-
'''
RNN encoder-decoder architecture
for statistical machine translation
'''

from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import lasagne


class RNNEncDecAttnModel:
    def __init__(self, cp):
	print("Building network ...")
       
	# First, we build the network, starting with an input layer
	# Recurrent layers expect input of shape
	# (batch size, SEQ_LENGTH, num_features)

	l_in = lasagne.layers.InputLayer(shape=(None, None, 30000))
	'''

	# We now build the LSTM layer which takes l_in as the input layer
	# We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients. 

	l_forward_1 = lasagne.layers.LSTMLayer(
	    l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
	    nonlinearity=lasagne.nonlinearities.tanh)

	l_forward_2 = lasagne.layers.LSTMLayer(
	    l_forward_1, N_HIDDEN, grad_clipping=GRAD_CLIP,
	    nonlinearity=lasagne.nonlinearities.tanh)

	# The l_forward layer creates an output of dimension (batch_size, SEQ_LENGTH, N_HIDDEN)
	# Since we are only interested in the final prediction, we isolate that quantity and feed it to the next layer. 
	# The output of the sliced layer will then be of size (batch_size, N_HIDDEN)
	l_forward_slice = lasagne.layers.SliceLayer(l_forward_2, -1, 1)

	# The sliced output is then passed through the softmax nonlinearity to create probability distribution of the prediction
	# The output of this stage is (batch_size, vocab_size)
	l_out = lasagne.layers.DenseLayer(l_forward_slice, num_units=vocab_size, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

	# Theano tensor for the targets
	target_values = T.ivector('target_output')
	
	# lasagne.layers.get_output produces a variable for the output of the net
	network_output = lasagne.layers.get_output(l_out)

	# The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
	cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()
	'''



    def train(self):
        pass

'''
how to implement maxout layer:

l1a = lasagne.layers.DenseLayer(l_in, nonlinearity=None, num_units=512, ...)
l1 = lasagne.layers.FeaturePoolLayer(l1a, ds=2)

not sure how many units to use or what to connect this to
'''
