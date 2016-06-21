# -*- coding: utf-8 -*-
'''
Prepare data
'''

import string
import numpy as np
import random

MAX_SENTENCE_LEN = 30
VOCAB_SIZE = 30000

np.random.seed(1337)

class CorpusPair:
    '''
    After initialization, contains a dictionary of most frequently 
    occuring VOCAB_SIZE words. Then, batches can be generated.
    '''

    current_offset = 0   # updated after batching
    num_sentences = 0    # updated when word matrices are built

    def __init__(self, source, dst):

        # open input files, store lines
        print "Parse training data"
        try:
            with open(source) as s, open(dst) as t:
                self.source = s.readlines()
                self.dst = t.readlines()
        except:
            print("Could not open input files")
            exit(1)

        # then preprocess 
        print " ... shuffle"
        self.shuffle()
        print " ... build lookup maps"
        self.build_dicts()
        print " ... vectorize data\n"
        self.vectorize_corpora()


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
        self.vec_src = np.zeros(shape=(self.num_sentences,MAX_SENTENCE_LEN,VOCAB_SIZE,))
        self.vec_dst = np.zeros(shape=(self.num_sentences,MAX_SENTENCE_LEN,VOCAB_SIZE,))

        # Vectorize all sentences of valid length
        def encode(idx, idy, word, word_dict, sen_arr):
            if word in word_dict[1]:
                sen_arr[idx,idy,word_dict[1][word]] = 1
            else:
                sen_arr[idx,idy,0] = 1

        idx = 0
        for idl, line in enumerate(self.source):
            if words_in_s(self.source[idl]) <= MAX_SENTENCE_LEN \
                    and words_in_s(self.dst[idl]) <= MAX_SENTENCE_LEN:
                for idw, word in enumerate(self.source[idl].split()):
                    encode(idx, idw, word, self.words_src, self.vec_src)
                for idw, word in enumerate(self.dst[idl].split()):
                    encode(idx, idw, word, self.words_dst, self.vec_dst)
                idx += 1


    def decode(self, sentence, src=True):
        '''
        Given an encoded sentence matrix,
        return the represented sentence string (tokenized).
        '''

        words = []

        for word in sentence:
            idxs = np.nonzero(word)[0]
            if len(idxs) > 1:
                raise Exception("Multiple hot bits on word vec")
            elif len(idxs) == 0:
                continue

            if src:
                words.append(self.words_src[0][idxs[0]])
            else:
                words.append(self.words_dst[0][idxs[0]])

        return ' '.join(words)

    def decode_src(self, sentence):
        return self.decode(sentence)

    def decode_dst(self, sentence):
        return self.decode(sentence, src=False)



    def get_minibatches(self):
        '''
        Shuffle, retrieve 1600 sentence pairs
        Sort by length, split into 20 minibatches
        '''
        print "Generating batch at i =",self.current_offset

        def sentence_len(sentence):
            length = 0
            for word in sentence:
                if len(np.nonzero(word)[0]) == 1:
                    length += 1
                else:
                    break
            return length

        # Select 1600 sentences
        src_pairs = self.vec_src[self.current_offset:self.current_offset+1600]
        dst_pairs = self.vec_dst[self.current_offset:self.current_offset+1600]
        self.current_offset += 1600

        if self.current_offset > self.num_sentences:
            self.current_offset -= self.num_sentences

        # Sort by sentence length
        # TODO Speed up this list comprehension
        lengths = [sentence_len(sentence) for sentence in src_pairs]
        z = list(zip(lengths, src_pairs, dst_pairs))
        z = sorted(z, key=lambda x: x[0])
        _, src_pairs, dst_pairs = zip(*z)

        # Split into 20 minibatches
        batches = []
        for i in range(1,21):
            batches.append((src_pairs[i*0:i*80], dst_pairs[i*0:i*80]))

        return batches
