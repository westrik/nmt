# -*- coding: utf-8 -*-
'''
Encoder-decoder architecture
for statistical machine translation
'''

from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import lasagne


LEARNING_RATE = .001
NUM_EPOCHS = .001
