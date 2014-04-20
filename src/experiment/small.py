#! /usr/bin/env python
'''
Small experiment: learn a small set of random dictionaries using all selection policies.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from data.generator import FromDictionaryL1
from data.dictionary import RandomGabors
from algorithms.selection import Unif, MagS, MagD, MXGS, MXGD
from algorithms.encoding import LASSO, KSparse
from algorithms.updating import InverseEta, GD
from inc.design import Experiment

def create(name):
    true_dictionary = RandomGabors(p = 6, K = 36)
    selectors = [cls(100) for cls in [Unif, MagS, MagD, MXGS, MXGD]]
    plambda = .1
    encoders = [LASSO(plambda)]
    updater = GD(encoders[0], eta = InverseEta(half_life = 200))
    return Experiment(name, FromDictionaryL1(true_dictionary, lambdaS = plambda),
                            selectors = selectors, encoders = encoders, updaters = [updater])
