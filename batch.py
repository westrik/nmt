# -*- coding: utf-8 -*-
'''
Prepare data
'''

import string
import numpy as np

MAX_SENTENCE_LEN = 30
VOCAB_SIZE = 30000

class CorpusPair:
    '''
    After initialization, contains a dictionary of most frequently 
    occuring VOCAB_SIZE words. Then, batches can be generated.
    '''

    words_src = None
    words_dst = None
    source =    None
    dst =       None


    def __init__(self, source, dst):
        # open input files, store lines
        try:
            with open(source) as s, open(dst) as t:
                self.source = s.readlines()
                self.dst = t.readlines()
        except:
            print("Could not open input files")
            exit(1)

        # then preprocess 
        self.count_word_usage()
        self.vectorize_corpora()


    def count_word_usage(self):
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
        num_sentences = 0
        for idx,line in enumerate(self.source):
            if words_in_s(self.source[idx]) <= MAX_SENTENCE_LEN \
                    and words_in_s(self.dst[idx]) <= MAX_SENTENCE_LEN:
                num_sentences += 1

        # Initialize numpy matrices to store sentences as vectors
        sen_src = np.zeros(shape=(num_sentences,30,VOCAB_SIZE,))
        sen_dst = np.zeros(shape=(num_sentences,30,VOCAB_SIZE,))

        # Vectorize all sentences of valid length
        def encode(idx, word, word_dict, sen_arr):
            for idy, word in enumerate(line.split(' ')):
                if word in word_dict[1]:
                    sen_arr[idx,idy,word_dict[1][word]] = 1
                else:
                    sen_arr[idx,idy,0] = 1

        for idx, line in enumerate(self.source):
            if words_in_s(self.source[idx]) <= MAX_SENTENCE_LEN \
                    and words_in_s(self.dst[idx]) <= MAX_SENTENCE_LEN:
                for idw, word in enumerate(self.source[idx].split()):
                    encode(idw, word, self.words_src, sen_src)
                for idw, word in enumerate(self.dst[idx].split()):
                    encode(idw, word, self.words_dst, sen_dst)


    def shuffle_and_split(self):
        '''
        Shuffle, retrieve 1600 sentence pairs
        Sort by length, split into 20 minibatches
        '''
        pass
