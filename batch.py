# -*- coding: utf-8 -*-
'''
Prepare data
'''

class CorpusPair:
    '''
    After initialization, contains a dictionary of most frequently 
    occuring 30k words, and smaller batches of the original dataset.
    '''

    words = []
    batches = []
    source = None
    target = None

    def __init__(self, source, target):
        self.source = source
        self.target = target

        self.prepare_data()
        self.split_into_batches()

    def prepare_data(self):
        '''
        Parse input files, create dictionary of most frequently used 
        30k words
        '''
        pass

    def split_into_batches(self):
        '''
        Using method from 
        '''
        pass
