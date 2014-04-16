'''
The learning algorithms update the dictionary (A) from the observables (X) and activations (S).

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy import power
from numpy.testing import assert_allclose, assert_equal
from data.dictionary import normalize

from algorithms.encoding import KSparse
from common import ary

class Base(object):
    def __init__(self, encoder, num_iter = 100, **kwds):
        self.encoder = encoder
        self.num_iter = num_iter
    

class GD(Base):
    """Update dictionaries using a simple (stochastic) gradient descent method.
    
    >>> encoder = KSparse(); updater = GD(encoder, eta = .01)
    
    >>> assert_equal(updater.eta, .01)
    
    >>> assert_equal(updater.encoder, encoder)
    """

    def __init__(self, encoder, eta = .1, **kwds):
        self.eta = eta
        super(GD, self).__init__(encoder, **kwds)
    
    def update(self, X, A):
        K = A.shape[1]
        for _ in range(self.num_iter):
            S = self.encoder.encode(X, A)
            Xr= A*S
            Gr= (Xr-X) * S.T / S.shape[1]
            A = A - self.eta / K * Gr
            A = ary(normalize(A))

        return asmatrix(A)

class KSVD(Base):
    """Update dictionaries using k-SVD method"""
    
    def __init__(self, encoder, **kwds):
        super(KSVD, self).__init__(encoder, **kwds)
    
    def update(self, X, A):
        # TODO write this
        return ary(A)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
