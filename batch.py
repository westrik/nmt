# -*- coding: utf-8 -*-
'''
Prepare data
'''

import string

class CorpusPair:
    '''
    After initialization, contains a dictionary of most frequently 
    occuring 30k words. Then, batches can be generated.
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

        print self.words_src[0]
        print self.words_src[1]


    def count_word_usage(self):
        '''
        Parse input files, create dictionary of most frequently used 30k words
        '''

        def build_count_dict(x):
            words = string.join(x).split(' ')
            freqs = {}
            for w in words:
                if w in freqs:
                    freqs[w] += 1
                else:
                    freqs[w] = 1

            aux = [(freqs[key], key) for key in freqs]
            aux.sort()
            aux.reverse()
            aux = [w[1] for w in aux][:30000]

            idx2w = {}
            w2idx = {}

            for idx, w in enumerate(aux):
                idx2w[idx] = w
                w2idx[w] = idx

            return (idx2w,w2idx)

        # Create list of top words
        # self.words_xxx[0] = index to word map
        # self.words_xxx[1] = word to index map
        self.words_src = build_count_dict(self.source)
        self.words_dst = build_count_dict(self.dst)

    def vectorize_corpora(self):
        '''
        Convert sentence pairs into vector representations
        '''

        # make a numpy array 
        pass


    def shuffle_and_split(self):
        '''
        Shuffle, retrieve 1600 sentence pairs
        Sort by length, split into 20 minibatches
        '''
        pass
