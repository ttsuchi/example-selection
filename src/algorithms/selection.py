'''
Example selection algorithms use the current dictionary (A) to select a number of "good" examples from a large set of observables.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy.random import permutation
from numpy.testing import assert_allclose

class Base(object):
    def __init__(self, n, **kwds):
        self.n = n
    
    @property
    def name(self):
        return self.__class__.__name__

class Unif(Base):
    """Randomly select examples
    
    >>> from numpy.random import randn
    
    >>> Unif(3).select(randn(2, 100), randn(2,3), randn(3, 100)).shape
    (3,)

    """
    def select(self, X, A, S):
        N=X.shape[1]
        return permutation(N)[:self.n]

def select_by_sum(G):
    """Return indices from a goodness matrix G, ordered descending for the sum across each example.

    >>> G=matrix([[.5,.4,1,0], [0,.9,.8,.7]])

    Since the sum is [.5, 1.3, 1.8, .7], 
    
    >>> print select_by_sum(G)
    [2 1 3 0]
    """
    return squeeze(asarray(argsort(-sum(G,axis=0))))

def select_per_dictionary(G):
    """Return indices from a goodness matrix G, ordered descending for each dictionary element.
    
    >>> G=matrix([[.5,.4,1,0], [0,.9,.8,.7]])
    
    Note that the ranking for each dictionary element is [2,0,1,3],[1,2,3,0]; so
    
    >>> print select_per_dictionary(G)
    [2 1 0 3]
    """
    R=argsort(-G) # Sort in descending order, with axis=1
    Rf=squeeze(asarray(R.T.flatten())) # Get indices column-wise
    _,idx=unique(Rf, return_index=True)
    return Rf[sort(idx)]
    

class MagS(Base):
    def select(self, X, A, S):
        return select_by_sum(abs(S))[:self.n]

class MagD(Base):
    def select(self, X, A, S):
        return select_per_dictionary(abs(S))[:self.n]
    
class MXGS(Base):
    def select(self, X, A, S):
        G=abs(multiply(sum(A*S-X,axis=0),S)) # Auto-broadcasted to the shape of S
        return select_by_sum(G)[:self.n]
    
class MXGD(Base):
    def select(self, X, A, S):
        G=abs(multiply(sum(A*S-X,axis=0),S)) # Auto-broadcasted to the shape of S
        return select_per_dictionary(G)[:self.n]

if __name__ == '__main__':
    import doctest
    doctest.testmod()
