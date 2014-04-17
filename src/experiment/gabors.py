#! /usr/bin/env python
'''
Experiment 1: learn Gabor filter-generated samples.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from data.observable import FromDictionaryL1
from data.dictionary import RandomGabors

from algorithms.selection import Unif, MagS, MagD, MXGS, MXGD
from algorithms.encoding import KSparse
from algorithms.updating import GD
from inc.design import Experiment

def create(name):
    true_dictionary = RandomGabors(p = 6, K = 13*13)
    selectors = [cls(100) for cls in [Unif, MagS, MagD, MXGS, MXGD]]
    encoder = KSparse(K = 3)
    updater = GD(encoder)
    return Experiment(name, FromDictionaryL1(true_dictionary),
                            selectors = selectors, encoders = [encoder], updaters = [updater])

