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

BATCH_SIZE = 30
NUM_EPOCH = 4

class RNNEncDecModel:
    def __init__(self, cp):
	print("Building network ...")

	# First, we build the network, starting with an input layer
	# Recurrent layers expect input of shape
	# (batch size, SEQ_LENGTH, num_features)

	# this is the placeholder tensor for the input sequences
	sequence = Input(shape=(maxlen,), dtype='int32')
	# this embedding layer will transform the sequences of integers
	# into vectors of size 128
	embedded = Embedding(max_features, 128, input_length=maxlen)(sequence)

	# apply forwards LSTM
	forwards = LSTM(64)(embedded)
	# apply backwards LSTM
	backwards = LSTM(64, go_backwards=True)(embedded)

	# concatenate the outputs of the 2 LSTMs
	merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
	after_dp = Dropout(0.5)(merged)
	output = Dense(1, activation='sigmoid')(after_dp)

	self.model = Model(input=sequence, output=output)

	# try using different optimizers and different optimizer configs
	self.model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

		
    def train(self):
	self.model.fit(self.X_train, self.y_train,
	      batch_size=BATCH_SIZE,
	      nb_epoch=NUM_EPOCH,
	      validation_data=[self.X_test, self.y_test])
