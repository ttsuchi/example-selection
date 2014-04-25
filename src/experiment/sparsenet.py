#! /usr/bin/env python
'''
Learn a small set of dictionaries from the same dataset used in SPARSENET.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from data.generator import FromImageDataset
from algorithms.selection import ALL_SELECTORS
from algorithms.encoding import LASSO, KSparse
from algorithms.updating import GD, SPAMS
from inc.design import Experiment

def create(name):
    selectors = [cls(400) for cls in ALL_SELECTORS]
    plambda = 0.15
    encoders = [LASSO(plambda)]
    updater = SPAMS(encoders[0])
    return Experiment(name, FromImageDataset('../contrib/sparsenet/IMAGES.mat', p = 16, K = 192),
                            selectors = selectors, encoders = encoders, updaters = [updater])
