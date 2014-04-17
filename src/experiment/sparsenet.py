#! /usr/bin/env python
'''
Learn a small set of dictionaries from the same dataset used in SPARSENET.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from data.generator import FromImageDataset
from algorithms.selection import Unif, MagS, MagD, MXGS, MXGD
from algorithms.encoding import LASSO, KSparse
from algorithms.updating import GD
from inc.design import Experiment

def create(name):
    selectors = [cls(100) for cls in [MXGD]]
    plambda = 0.1
    encoders = [LASSO(plambda)]
    updater = GD(encoders[0])
    return Experiment(name, FromImageDataset('../contrib/sparsenet/IMAGES.mat', p = 16, K = 192),
                            selectors = selectors, encoders = encoders, updaters = [updater])
