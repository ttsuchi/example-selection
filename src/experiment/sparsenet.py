#! /usr/bin/env python
'''
Learn a small set of dictionaries from the same dataset used in SPARSENET.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from data.generator import FromImageDataset
from algorithms.selection import *
from algorithms.encoding import LASSO, KSparse, SOMP
from algorithms.updating import GD, SPAMS
from inc.design import Experiment

def create(name):
    selectors = [cls(400) for cls in [Unif]]
    plambda = 0.15
    encoders = [LASSO(plambda), SOMP()]
    updater = [GD(encoders[0]), SPAMS()]
    return Experiment(name, FromImageDataset('../contrib/sparsenet/IMAGES.mat', p = 12, K = 144),
                            selectors = selectors, encoders = encoders, updaters = updater)
