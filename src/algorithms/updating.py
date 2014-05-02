'''
The learning algorithms update the dictionary (A) from the observables (X) and activations (S).

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy import power
from numpy.testing import assert_allclose, assert_equal
from data.dictionary import normalize

from algorithms.encoding import equalize_activities, KSparse, SOMP
from analyses.stats import collect_stats
from inc.common import mtr

from spams import trainDL

def update_with(design, X, Sstar, Xsnr, A, itr):
    """Return a new dictionary using the examples picked by the current selection policy.
    """
    all_stats = {}
    
    # Encode all training examples
    S = design.encoder.encode(X, A)
    
    # Pick examples to learn from
    idx = design.selector.select(X, A, S)
    
    # Update dictionary using these examples
    Xp = mtr(X[:, idx])
    newA = design.updater.update(Xp, A, itr)

    # Collect the stats (and A will be re-ordered)
    stats, A = collect_stats(X, newA, A, design.experiment.Astar, S, Sstar, Xsnr, idx)
    all_stats.update(stats)

    # Some top chosen examples
    Xp = X[:, idx[:min(len(idx), A.shape[1])]]
    return A, all_stats, Xp

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

class _Base(object):
    def __init__(self, encoder, num_iter = 100, **kwds):
        self.encoder = encoder
        self.num_iter = num_iter
    

class GD(_Base):
    """Update dictionaries using a simple (stochastic) gradient descent method.
    
    >>> encoder = KSparse(); updater = GD(encoder, eta = .01)
    
    >>> assert_equal(updater.eta, .01)
    
    >>> assert_equal(updater.encoder, encoder)
    """

    def __init__(self, encoder, eta = InverseEta(), eq_power = .5,  **kwds):
        self.eta = eta
        self.eq_power = eq_power
        super(GD, self).__init__(encoder, **kwds)
    
    def update(self, X, A, itr):
        K = A.shape[1]
        for _ in range(self.num_iter):
            S = self.encoder.encode(X, A)
            S = equalize_activities(S, self.eq_power)
            Xr= A*S
            Gr= (Xr-X) * S.T / S.shape[1]
            eta = self.eta(itr) if hasattr(self.eta, '__call__') else self.eta
            A = A - eta / K * Gr
            A = mtr(normalize(A))

        return A

class SPAMS(_Base):
    """Learns the dictionary using the SPAMS package.
    """
    def __init__(self, encoder, lambda1 = .15, **kwds):
        super(SPAMS, self).__init__(encoder, **kwds)
            
        if encoder.__class__ == SOMP:
            # Use the SOMP here too
            self.param = {
                'mode': 4,
                'lambda1':  encoder.nnz
            }
        
        elif hasattr(encoder, 'spams_param'):
            self.param = encoder.spams_param.copy()
            del self.param['L']
            del self.param['pos']
        else:
            self.param = {
                'mode': 2,
                'lambda1':  lambda1,
                'lambda2':  0
            }

        self.param.update({
            'posAlpha': True,
            'clean':    False,
            'iter':     self.num_iter,
            'verbose':  False
            })
        
        print self.param
        self.model = None
    
    def update(self, X, A, itr):
        param = {
          'D': A,
          'batchsize': X.shape[1]
        }
        param.update(self.param)
        
        if self.model is None:
            A, self.model = trainDL(X, return_model = True, **param)
        else:
            A, self.model = trainDL(X, return_model = True, model = self.model, **param)
        
        return mtr(normalize(A))

class KSVD(_Base):
    """Update dictionaries using k-SVD method"""
    
    def __init__(self, encoder, **kwds):
        super(KSVD, self).__init__(encoder, **kwds)
    
    def update(self, X, A, itr):
        # TODO write this
        return mtr(A)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
