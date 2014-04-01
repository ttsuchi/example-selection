'''
Generates an arbitrary number of training examples based on randomly-generated dictionary elements.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt
import time

# From http://xcorr.net/2011/05/27/whiten-a-matrix-matlab-code/
def whiten(X):
    fudgefactor=1e-12;
    X = asmatrix(X - mean(X))
    D, V = eig(X * X.T)
    return asarray(V.T * power(diag(1 / (D + fudgefactor)), .5) * V * X)

# A = X'*X;
# [V,D] = eig(A);
# X = X*V*diag(1./(diag(D)+fudgefactor).^(1/2))*V';


class Observables:
    '''
        Generates observable signals X from a set of known dictionaries A.
    '''
    def __init__(self, A):
        self.A = A

    def generate(self, N):
        X = self.generate_raw(N)
        X = self.whiten(X)
    

class Random:
    '''
        Creates a random dictionary, each with P dimensions and K elements.
    '''
    def __init__(self, P = 128, K = 512, lambdaA = 1, lambdaS = 1, sigma = .1):
        self.P = P
        self.K = K
        self.lambdaA = lambdaA
        self.A = -log(random.random((self.P, self.K)))/self.lambdaA * ((random.random((self.P, self.K)) > .5)*2-1)
        self.A /= sqrt(dot(self.A.T, self.A).diagonal())
        
        # For generating examples
        self.lambdaS = lambdaS
        self.sigma = sigma

    def generate(self,N):
        # Generate new set of examples
        S=-log(random.random((self.K, self.N)))/self.lambdaS
        return dot(self.A,S)+random.standard_normal((self.P, self.N))*self.sigma
