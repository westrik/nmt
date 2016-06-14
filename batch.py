# -*- coding: utf-8 -*-
'''
Prepare data
'''

import string
import numpy as np
import random

MAX_SENTENCE_LEN = 30
VOCAB_SIZE = 30000

class CorpusPair:
    '''
    After initialization, contains a dictionary of most frequently 
    occuring VOCAB_SIZE words. Then, batches can be generated.
    '''

    def __init__(self, source, dst):
        self.num_sentences = 0

        # open input files, store lines
        try:
            with open(source) as s, open(dst) as t:
                self.source = s.readlines()
                self.dst = t.readlines()
        except:
            print("Could not open input files")
            exit(1)

        # then preprocess 
        self.shuffle()
        self.build_dicts()
        self.vectorize_corpora()

        self.decode(self.vec_src[0])


    def shuffle(self):
        '''
        Prior to training, shuffle training data
        '''
        z = list(zip(self.source, self.dst))

        random.shuffle(z)

        self.source, self.dst = zip(*z)


    def build_dicts(self):
        '''
        Parse input files, create dictionary of most frequently used 30k words
        '''

        def build_count_dict(x):
            # Count frequencies of all words
            words = string.join(x).split(' ')
            freqs = {}
            for w in words:
                if w in freqs:
                    freqs[w] += 1
                else:
                    freqs[w] = 1

            # Sort words by frequency
            aux = [(freqs[key], key) for key in freqs]
            aux.sort()
            aux.reverse()
            aux = [w[1] for w in aux][:VOCAB_SIZE-1]

            # Build bidirectional dictionary to store word mappings
            idx2w = {}
            w2idx = {}
            for idx, w in enumerate(aux):
                idx2w[idx+1] = w
                w2idx[w] = idx+1

            # Store mapping for unknown words
            idx2w[0] = "[UNK]"
            w2idx["[UNK]"] = 0

            return (idx2w, w2idx)

        # Create list of top words
        # self.words_xxx[0] = index to word map
        # self.words_xxx[1] = word to index map
        self.words_src = build_count_dict(self.source)
        self.words_dst = build_count_dict(self.dst)


    def vectorize_corpora(self):
        '''
        Convert sentence pairs into vector representations
        '''

        def words_in_s(s): return len(s.split())

        # Count sentences that fit within length limit
        for idx,line in enumerate(self.source):
            if words_in_s(self.source[idx]) <= MAX_SENTENCE_LEN \
                    and words_in_s(self.dst[idx]) <= MAX_SENTENCE_LEN:
                self.num_sentences += 1

        # Initialize numpy matrices to store sentences as vectors
        self.vec_src = np.zeros(shape=(self.num_sentences,30,VOCAB_SIZE,))
        self.vec_dst = np.zeros(shape=(self.num_sentences,30,VOCAB_SIZE,))

        # Vectorize all sentences of valid length
        def encode(idx, idy, word, word_dict, sen_arr):
            if word in word_dict[1]:
                sen_arr[idx,idy,word_dict[1][word]] = 1
            else:
                sen_arr[idx,idy,0] = 1

        idx = 0
        for line in enumerate(self.source):
            if words_in_s(self.source[idx]) <= MAX_SENTENCE_LEN \
                    and words_in_s(self.dst[idx]) <= MAX_SENTENCE_LEN:
                for idw, word in enumerate(self.source[idx].split()):
                    encode(idx, idw, word, self.words_src, self.vec_src)
                for idw, word in enumerate(self.dst[idx].split()):
                    encode(idx, idw, word, self.words_dst, self.vec_dst)
                idx += 1


    def decode(self, sentence, src=True):
        '''
        Given an encoded sentence matrix,
        return the represented sentence string (tokenized).
        '''
        for word in sentence:
            print word
            #print np.nonzero()


    def decode_src(self, sentence):
        return self.decode(sentence)

    def decode_dst(self, sentence):
        return self.decode(sentence, src=False)



    def get_minibatches(self):
        '''
        Shuffle, retrieve 1600 sentence pairs
        Sort by length, split into 20 minibatches
        '''

        def sentence_len(sentence):
            pass

        idxs = random.sample(range(0, self.num_sentences), 1600)
