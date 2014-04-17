'''
The learning algorithms update the dictionary (A) from the observables (X) and activations (S).

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy import power
from numpy.testing import assert_allclose, assert_equal
from data.dictionary import normalize

from algorithms.encoding import KSparse
from inc.common import mtr

class InverseEta(object):
    """A function object that produces learning rates that decay with O(t^-1).

    >>> eta = InverseEta(vmax = .1, vmin = .001, half_life = 200)
    
    >>> assert_allclose(.1, eta(0)); assert_allclose(.001, eta(1e12))

    >>> assert_allclose(.001 + (.1 - .001) / 2, eta(200))
    """
    def __init__(self, vmax = .1, vmin = 0, half_life = 200):
        self.vmax = vmax
        self.vmin = vmin
        self.half_life = float(half_life)

    def __call__(self, itr):
        return (self.vmax - self.vmin) / (1 + float(itr) / self.half_life) + self.vmin

class ExpDecayEta(object):
    """A function object that produces exponentially decaying eta with respect to the iterations.
    
    >>> eta = ExpDecayEta(vmax = .1, vmin = .001, half_life = 200)
    
    >>> assert_allclose(.1, eta(0)); assert_allclose(.001, eta(10000))

    >>> assert_allclose(.001 + (.1 - .001) / 2, eta(200))

    """
    def __init__(self, vmax = .1, vmin = .001, half_life = 200):
        self.vmax = vmax
        self.vmin = vmin
        self.half_life = float(half_life)
    
    def __call__(self, itr):
        return self.vmin + power(2, -float(itr)/self.half_life) * (self.vmax - self.vmin)

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

    def __init__(self, encoder, eta = InverseEta(), **kwds):
        self.eta = eta
        super(GD, self).__init__(encoder, **kwds)
    
    def update(self, X, A, itr):
        K = A.shape[1]
        for _ in range(self.num_iter):
            S = self.encoder.encode(X, A)
            Xr= A*S
            Gr= (Xr-X) * S.T / S.shape[1]
            eta = self.eta(itr) if hasattr(self.eta, '__call__') else self.eta
            A = A - eta / K * Gr
            A = mtr(normalize(A))

        return A

class KSVD(Base):
    """Update dictionaries using k-SVD method"""
    
    def __init__(self, encoder, **kwds):
        super(KSVD, self).__init__(encoder, **kwds)
    
    def update(self, X, A, itr):
        # TODO write this
        return mtr(A)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
