'''
Created on Mar 25, 2014

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''

from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt
import time
import data.synthetic_random

class Dictionaries:   
    def __init__(self):
        self.do_plot = True
        self.generator = generator
        self.P = 128
        self.K = 64
        
        self.lambdaA   = 1
        self.lambdaS   = 1
        self.sigma     = .1
        self.sigma_sal = .1
        
        self.N = 10000
        self.Np = self.K*10
        
        # Learning rate
        self.eta = .01/self.K
        
        # Number of iterations
        self.TL=15
        self.TS=10
        
        # Sampling policies
        self.policies=[self.unif, self.sala, self.salc, self.mxga, self.mxgc]
        self.NP=len(self.policies)
        self.policy_names=map(lambda f: f.__name__, self.policies)
        
        # Generate random dictionary elements
        self.A = -log(random.random((self.P, self.K)))/self.lambdaA * ((random.random((self.P, self.K)) > .5)*2-1)
        self.A /= sqrt(dot(self.A.T, self.A).diagonal())

    def learn(self, data, encoder, selector):