# -*- coding: utf-8 -*-
'''
RNN encoder-decoder model
for statistical machine translation

Based on model from Sutskever et al 2014
https://arxiv.org/pdf/1409.3215v3.pdf
'''

from __future__ import print_function

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

np.random.seed(1337)

class RNNEncDecModel:
    def __init__(self, cp):
	print("Building network ...")
       

	# gated hidden unit with GRU or LSTM
	model = Sequential()

	model.add(Dense(output_dim=64, input_dim=100))
	model.add(Activation("relu"))
	model.add(Dense(output_dim=10))
	model.add(Activation("softmax"))

	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

	# First, we build the network, starting with an input layer
	# Recurrent layers expect input of shape
	# (batch size, SEQ_LENGTH, num_features)


    def train(self):
        pass

