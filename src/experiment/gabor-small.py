#! /usr/bin/env python
'''
Experiment 1: learn Gabor filter-generated samples.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from data.generator import FromDictionaryL1
from data.dictionary import RandomGabors
from algorithms.selection import Unif, UsedD, MagS, MagD, MXGS, MXGD, SalMap
from algorithms.encoding import LASSO, KSparse
from algorithms.updating import GD
from inc.design import Experiment

def create(name):
    K = 9
    true_dictionary = RandomGabors(p = 12, K = K)
    selectors = [cls(K * 10) for cls in [Unif, UsedD, MXGS, SalMap]]
    encoder = KSparse(2)
    updater = GD(encoder, eq_power = .8)
    return Experiment(name, FromDictionaryL1(true_dictionary, lambdaS = 1, snr = 1),
                            selectors = selectors, encoders = [encoder], updaters = [updater])
