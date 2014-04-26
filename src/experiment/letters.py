#! /usr/bin/env python
'''
Experiment 1: learn Gabor filter-generated samples.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from data.generator import *
from data.dictionary import Letters
from algorithms.selection import *
from algorithms.encoding import LASSO, KSparse
from algorithms.updating import GD, SPAMS
from inc.design import Experiment

def create(name):
    K = 25
    true_dictionary = Letters(p = 8, K = K)
    selectors = [cls(K * 10) for cls in ALL_SELECTORS]
    encoder = LASSO(.15)
    updaters = [GD(encoder, num_iter = 500), SPAMS(encoder, num_iter = 10)]
    return Experiment(name, FromDictionaryL0(true_dictionary, nnz = 3, snr = 6),
                            selectors = selectors, encoders = [encoder], updaters = updaters)

