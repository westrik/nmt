# -*- coding: utf-8 -*-
'''
Train model
'''

from batch import CorpusPair
from model import RNNEncDecAttnModel


DEV_SRC = '../datasets/translation/dev-set/newstest-2012+2013.tok.en'
DEV_DST = '../datasets/translation/dev-set/newstest-2012+2013.tok.fr'


def main():
    ''' 
    - Open input files for x and y
    - Prep & split into batches
    - Train on each batch
    - Save trained model
    '''

    cp = CorpusPair(DEV_SRC, DEV_DST)
    model = RNNEncDecAttnModel()

    for b in cp.batches:
        model.train(b)


if __name__ == "__main__":
    main()
