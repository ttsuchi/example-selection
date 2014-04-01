'''
The learning algorithms update the dictionary (A) from the observables (X) and activations (S).

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy import power
from numpy.testing import assert_allclose
from data.dictionary import normalize

class Base(object):
    def __init__(self, encoding_fn, num_iter = 100, **kwds):
        self.encoding_fn = encoding_fn
        self.num_iter = num_iter
    

class GD(Base):
    """Update dictionaries using a simple (stochastic) gradient descent method."""

    def __init__(self, eta = .1, **kwds):
        self.eta = eta
        super(GD, self).__init__(**kwds)
    
    def update(self, X, A):
        K = A.shape[1]
        for _ in range(self.num_iter):
            S = self.encoding_fn(X, A)
            Xr= A*S
            Gr= (Xr-X) * S.T / S.shape[1]
            A = A - self.eta / K * Gr
            A = normalize(A)

        return A

class KSVD(Base):
    """Update dictionaries using k-SVD method"""
    
    def __init__(self, **kwds):
        pass
    
    def update(self, X, A):
        # TODO write this
        return A

if __name__ == '__main__':
    import doctest
    doctest.testmod()
