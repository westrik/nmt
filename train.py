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
    - open input files for x and y
    - prep & split into batches
    - train on each batch
    '''

    cp = CorpusPair(DEV_SRC, DEV_DST)


if __name__ == "__main__":
    main()
