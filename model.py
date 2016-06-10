# -*- coding: utf-8 -*-
'''
Encoder-decoder architecture
for statistical machine translation
'''

# have a model that can be trained
# encoder:
# decoder:

from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import lasagne


class RNNEncDecAttnModel:
    pass


'''
how to implement maxout layer:

l1a = lasagne.layers.DenseLayer(l_in, nonlinearity=None, num_units=512, ...)
l1 = lasagne.layers.FeaturePoolLayer(l1a, ds=2)

not sure how many units to use or what to connect this to

'''
