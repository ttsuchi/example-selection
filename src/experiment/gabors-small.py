#! /usr/bin/env python
'''
Experiment 1: learn Gabor filter-generated samples.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from data.generator import FromDictionaryL0
from data.dictionary import RandomGabors
from algorithms.selection import ALL_SELECTORS
from algorithms.encoding import LASSO, KSparse
from algorithms.updating import GD, SPAMS
from inc.design import Experiment

def create(name):
    K = 25
    true_dictionary = RandomGabors(p = 12, K = K)
    selectors = [cls(K * 10) for cls in ALL_SELECTORS]
    encoder = LASSO(.15)
    updater = GD(encoder, num_iter = 10)
    return Experiment(name, FromDictionaryL0(true_dictionary, nnz = 2, snr = 6),
                            selectors = selectors, encoders = [encoder], updaters = [updater])

