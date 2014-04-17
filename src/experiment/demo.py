#! /usr/bin/env python
'''
Demo experiment: simple dictionary learning based on random dictionaries.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from data.observable import FromDictionaryL1
from data.dictionary import Random

from algorithms.selection import Unif, MXGD
from algorithms.encoding import KSparse
from algorithms.updating import GD
from inc.design import Experiment

def create(name):
    true_dictionary = Random(4, 32)
    selectors = [cls(100) for cls in [Unif, MXGD]]
    encoder = KSparse(K = 2)
    updater = GD(encoder)
    return Experiment(name, 
                      FromDictionaryL1(true_dictionary),
                      selectors = selectors, encoders = [encoder], updaters = [updater])
