#! /usr/bin/env python
'''
Experiment 1: learn Gabor filter-generated samples.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from data.generator import FromDictionaryL1
from data.dictionary import RandomGabors
from algorithms.selection import Unif, UsedD, MagS, MagD, MXGS, MXGD, SalMap
from algorithms.encoding import KSparse
from algorithms.updating import GD
from inc.design import Experiment

def create(name):
    true_dictionary = RandomGabors(p = 12, K = 12*12)
    selectors = [cls(200) for cls in [Unif, UsedD, MagS, MagD, MXGS, MXGD, SalMap]]
    encoder = KSparse(K = 3)
    updater = GD(encoder)
    return Experiment(name, FromDictionaryL1(true_dictionary),
                            selectors = selectors, encoders = [encoder], updaters = [updater])

