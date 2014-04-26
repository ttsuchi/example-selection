'''
Example selection algorithms use the current dictionary (A) to select a number of "good" examples from a large set of observables.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy.random import permutation
from numpy.testing import assert_allclose

import scipy.ndimage.filters

import Pycluster

import cv2 as cv

class _Base(object):
    def __init__(self, n, **kwds):
        self.n = n
    
    @property
    def name(self):
        return self.__class__.__name__

class Unif(_Base):
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

class UsedD(_Base):
    """Returns examples that use the dictionary element; similar to the algorithm used in K-SVD.
    For L1 activations, not all activations will be exactly zero. So will consider the dictionary to have been "used" if it's greater than the median.
    """
    def select(self, X, A, S):
        G = (S>median(S, axis=0))*1.0
        return select_per_dictionary(G)[:self.n]

class MagS(_Base):
    def select(self, X, A, S):
        return select_by_sum(abs(S))[:self.n]

class MagD(_Base):
    def select(self, X, A, S):
        return select_per_dictionary(abs(S))[:self.n]
    
class MXGS(_Base):
    def select(self, X, A, S):
        G=abs(multiply(sum(A*S-X,axis=0),S)) # Auto-broadcasted to the shape of S
        return select_by_sum(G)[:self.n]
    
class MXGD(_Base):
    def select(self, X, A, S):
        G=abs(multiply(sum(A*S-X,axis=0),S)) # Auto-broadcasted to the shape of S
        return select_per_dictionary(G)[:self.n]

class SalMap(_Base):
    """Simplified implementation of the saliency map selection.
    Since each example has a single channel (grayscale) and is very small, will only use the "intensity" and "orientation" channels at a single scale.
    """
    def __init__(self, n, **kwds):
        super(SalMap, self).__init__(n, **kwds)
        self.kernels = None
    
    def _normalize(self, M):
        """Implements the map normalization operator. Since the patches are small, we only perform the range normalization.
        """
        N = M - M.min()
        N = N / N.max()  # Normalize to [0,1]
        return N
    
    def select(self, X, A, S):
        P, N = X.shape
        p = int(sqrt(P))
        
        # "Intensity" is just the input
        I = self._normalize(abs(X))

        # Apply gabor filters to get the orientation channel
        if self.kernels is None:
            self.kernels = [cv.getGaborKernel((p, p), p / 2, pi * angle / 4, p, 1) for angle in [0, 45, 90, 135]]
        
        Xs = zeros((p, p*N))
        for n in range(N):
            Xs[:, p*n:(p*n+p)] = X[:,n].reshape((p,p))

        Os = self._normalize(reduce(add, [self._normalize(cv.filter2D(Xs, cv.CV_32F, kernel)) for kernel in self.kernels]))
   
        O = zeros((P, N))
        for n in range(N):
            O[:, n] = Os[:,p*n:(p*n+p)].reshape((P))
        
        G=.5 * (I + O)
        return select_by_sum(G)[:self.n]

class SUNS(_Base):
    """Simplified implementation of the saliency using natural statistics.
    Will assume exponential distribution on non-zero components. That means the saliency is simply s / mean(s) for each dimension.
    """
    
    def select(self, X, A, S):
        G = abs(S) / mean(abs(S), axis=1)
        return select_by_sum(G)[:self.n]

class SUND(_Base):    
    def select(self, X, A, S):
        G = abs(S) / mean(abs(S), axis=1)
        return select_per_dictionary(G)[:self.n]

class KMX(_Base):
    """Choose examples based on k-means cluster centroid distances of X
    """
    def select(self, X, A, S):
        _, K = A.shape
        labels, _, _ = Pycluster.kcluster(X.T, K)
        centers, _   = Pycluster.clustercentroids(X.T, clusterid=labels)
        centers = centers.T
        G = zeros(S.shape)
        
        for k in range(K):
            D = expand_dims(centers[:, k], axis=1)
            G[k, :] = 1/sqrt(sum(multiply(D, D), axis=0) + spacing(1))

        return select_per_dictionary(G)[:self.n]

class KMS(_Base):
    """Choose examples based on k-means cluster centroid distances of S
    """
    def select(self, X, A, S):
        _, K = A.shape
        labels, _, _ = Pycluster.kcluster(S.T, K)
        centers, _   = Pycluster.clustercentroids(S.T, clusterid=labels)
        centers = centers.T
        G = zeros(S.shape)
        
        for k in range(K):
            D = S - expand_dims(centers[:, k], axis=1)
            G[k, :] = 1/sqrt(sum(multiply(D, D), axis=0) + spacing(1))

        return select_per_dictionary(G)[:self.n]


ALL_SELECTORS = [Unif, UsedD, MagS, MagD, MXGS, MXGD, SalMap, SUNS, SUND, KMX, KMS]

if __name__ == '__main__':
    import doctest
    doctest.testmod()
