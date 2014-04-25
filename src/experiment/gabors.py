#! /usr/bin/env python
'''
Experiment 1: learn Gabor filter-generated samples.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from data.generator import FromDictionaryL1
from data.dictionary import RandomGabors
from algorithms.selection import ALL_SELECTORS
from algorithms.encoding import KSparse
from algorithms.updating import GD, SPAMS
from inc.design import Experiment

def create(name):
    true_dictionary = RandomGabors(p = 12, K = 12*12)
    selectors = [cls(200) for cls in ALL_SELECTORS]
    encoder = KSparse(K = 3)
    updater = SPAMS(encoder)
    return Experiment(name, FromDictionaryL1(true_dictionary),
                            selectors = selectors, encoders = [encoder], updaters = [updater])

