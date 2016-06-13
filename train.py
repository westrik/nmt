# -*- coding: utf-8 -*-
'''
Train model
'''

from batch import CorpusPair
from model import RNNEncDecAttnModel
import pickle as pkl


DEV_SRC = '../datasets/translation/dev-set/newstest-2012+2013.tok.en'
DEV_DST = '../datasets/translation/dev-set/newstest-2012+2013.tok.fr'


def main():
    ''' 
    - Open pretokenized input files for x and y
    - Prep & split into batches
    - Train on each batch
    - Save trained model
    '''

    cp = CorpusPair(DEV_SRC, DEV_DST)

    model = RNNEncDecAttnModel()


#    for x in cp.shuffle_and_split():
#        model.train(x)

    # dump bidir word dicts 
    with open('wordmap.pkl','w') as wordmap:
        pkl.dump((cp.words_src, cp.words_dst), wordmap)



if __name__ == "__main__":
    main()
