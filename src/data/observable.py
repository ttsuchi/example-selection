'''
Generates observable signals (X).

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy.linalg import eigh
from numpy.random import randn
from dictionary import Random

from numpy.testing import assert_allclose

def whiten(X):
    """Whiten the signal X so that it has zero mean and zero cross-correlation.

    From http://xcorr.net/2011/05/27/whiten-a-matrix-matlab-code/.

    >>> P=10; W=whiten(randn(P, 100))
    
    >>> assert_allclose(mean(W,axis=1), zeros((P,1)), atol=1e-5)
    
    >>> assert_allclose(var(W,axis=1),  ones((P,1)), atol=1e-5)
    
    >>> assert_allclose(cov(W)-diag(diag(cov(W))), zeros((P,P)), atol=1e-5)
    """
    fudgefactor=1e-18;
    X = asmatrix(X - asmatrix(mean(X,axis=1)).T)
    C = X * X.T / X.shape[1]
    d, V = eigh(C)
    D = asmatrix(diag(1./sqrt(d+fudgefactor)))
    return V * D * V.T * X

def snr_to_sigma(snr):
    """Convert noise SNR value to STD for noise generation, assuming the signal has unit variance
        (which the whiten() method does.)
        
    >>> assert_allclose(var(snr_to_sigma(.3)*randn(10000,1)), 1e-3, rtol=1e-1) # .3 dB
    
    """
    return 10**(-snr*5)

class Base(object):
    def __init__(self, N, **kwds):
        self.N = N # Number of samples to generate on each call

class FromDictionary(Base):
    """Generate examples from a set of "ground-truth" dictionary elements
    """
    def __init__(self, dictionary = Random, snr = 6, **kwds):
        self.dictionary = dictionary
        self.snr        = snr        # Signal-to-noise ratio in dB
        super(FromDictionary, self).__init__(**kwds)
    
    def generate(self):
        S = self.generate_S()
        X = self.A*S
        return whiten(X) + randn(X.shape[0], X.shape[1])*snr_to_sigma(self.snr)

class FromDictionaryL1(FromDictionary):
    """Generate examples from a set of "ground-truth" dictionary elements, using L1 sparsity
    """
    def __init__(self, lambdaS = 1, **kwds):
        self.lambdaS = lambdaS  # Sparsity
        super(FromDictionary, self).__init__(**kwds)
    
    def generate_S(self):
        return asmatrix(-log(randn(self.dictionary.K, self.N)) / self.lambdaS)

class FromImageDataset(Base):
    """Generate examples from image datasets.
    """
    def __init__(self, dataset_dir, p = 16, **kwds):       
        self.dataset_dir = dataset_dir
        self.p = p
        self.P = p*p
        super(FromImageDataset, self).__init__(**kwds)
    
    def generate(self):
        # TODO implement this
        X = zeros((self.P, self.N))
        return whiten(X)
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()
