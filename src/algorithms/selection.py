'''
Example selection algorithms use the current dictionary (A) to select a number of "good" examples from a large set of observables.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy.random import permutation, randn, randint
from numpy.testing import assert_allclose

import scipy.ndimage.filters
import Pycluster
import cv2 as cv

from data.dictionary import Random
from data.generator import FromDictionaryL1
import time
from operator import itemgetter


class _Base(object):
    def __init__(self, n, **kwds):
        self.n = n
    
    @property
    def name(self):
        return self.__class__.__name__

    def _select_by_sum(self, G):
        """Return indices from a goodness matrix G, ordered descending for the sum across each example.
    
        >>> G=matrix([[.5,.4,1,0], [0,.9,.8,.7]])
    
        Since the sum is [.5, 1.3, 1.8, .7], 
        
        >>> print _Base(4)._select_by_sum(G)
        [2 1 3 0]
        """
        return squeeze(asarray(argsort(-sum(G, axis=0))))[:self.n]

    def _select_per_dictionary(self, G):
        """Return indices from a goodness matrix G, ordered descending for each dictionary element.
        
        >>> G=matrix([[.5,.4,1,0], [0,.9,.8,.7]])
        
        Note that the ranking for each dictionary element is [2,0,1,3],[1,2,3,0]; so
        
        >>> print _Base(4)._select_per_dictionary(G)
        [2 1 0 3]
        """
        R = argsort(-G)  # Sort in descending order, with axis=1
        Rf = squeeze(asarray(R.T.flatten()))  # Get indices column-wise
        _, idx = unique(Rf, return_index=True)
        return (Rf[sort(idx)])[:self.n]

class Unif(_Base):
    """Randomly select examples
    
    >>> from numpy.random import randn
    
    >>> Unif(3).select(randn(2, 100), randn(2,3), randn(3, 100)).shape
    (3,)

    """
    def select(self, X, A, S):
        N = X.shape[1]
        return permutation(N)[:self.n]

    def group(self):
        return (0, 0)

def _is_used(S):
    return S > S.max(axis=0) * .8
    
class UsedD(_Base):
    """Returns examples that use the dictionary element; similar to the algorithm used in K-SVD.
    For L1 activations, not all activations will be exactly zero. So will consider the dictionary to have been "used" if it's greater than the mean.
    """
    def select(self, X, A, S):
        G = sum(_is_used(S), axis=0)
        return self._select_per_dictionary(G)

    def group(self):
        return (2, 0)

class MagS(_Base):
    def select(self, X, A, S):
        return self._select_by_sum(abs(S))

    def group(self):
        return (1, 0)

class MagD(_Base):
    def select(self, X, A, S):
        return self._select_per_dictionary(abs(S))

    def group(self):
        return (2, 0)

class MXGS(_Base):
    def select(self, X, A, S):
        G = abs(multiply(sum(A * S - X, axis=0), S))  # Auto-broadcasted to the shape of S
        return self._select_by_sum(G)

    def group(self):
        return (1, 0)
    
class MXGD(_Base):
    def select(self, X, A, S):
        G = abs(multiply(sum(A * S - X, axis=0), S))  # Auto-broadcasted to the shape of S
        return self._select_per_dictionary(G)

    def group(self):
        return (2, 0)

class SNRS(_Base):
    def select(self, X, A, S):
        E = X - A*S
        G = sum(multiply(X,X),axis=0) / (sum(multiply(E,E), axis=0)+spacing(1))
        return self._select_by_sum(G)

    def group(self):
        return (1, 0)

class SNRD(_Base):
    def select(self, X, A, S):
        E = X - A*S
        G = multiply((sum(multiply(X,X),axis=0) / (sum(multiply(E,E), axis=0)+spacing(1))), S)
        return self._select_per_dictionary(G)

    def group(self):
        return (2, 0)

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
            # only available on OpenCV 3+...
            if hasattr(cv, 'getGaborKernel'):
                self.kernels = [cv.getGaborKernel((p, p), p / 2, pi * angle / 4, p, 1) for angle in [0, 45, 90, 135]]
            else:
                import pickle
                with open("./algorithms/gabor-kernels-%d.pkl" % p, 'r') as fin:
                    self.kernels = pickle.load(fin)
        
        Xs = zeros((p, p * N))
        for n in range(N):
            Xs[:, p * n:(p * n + p)] = X[:, n].reshape((p, p))

        Os = self._normalize(reduce(add, [self._normalize(cv.filter2D(Xs, cv.CV_32F, kernel)) for kernel in self.kernels]))
   
        O = zeros((P, N))
        for n in range(N):
            O[:, n] = Os[:, p * n:(p * n + p)].reshape((P))
        
        G = .5 * (I + O)
        return self._select_by_sum(G)

    def group(self):
        return (0, 1)

class _SUN(_Base):
    """Simplified implementation of the saliency using natural statistics.
    Will assume exponential distribution on non-zero components. That means the saliency is simply s / mean(s) for each dimension.
    """
    def _G(self, X, A, S):
        G = abs(S.copy())
        G[G < mean(G) * .01] = nan
        m = nanmean(G, axis=1) + spacing(1)
        m[isnan(m)] = 1.0
        return S / m

class SUNS(_SUN):
    def select(self, X, A, S):
        return self._select_by_sum(self._G(X, A, S))

    def group(self):
        return (1, 1)

class SUND(_SUN):
    def select(self, X, A, S):
        return self._select_per_dictionary(self._G(X, A, S))

    def group(self):
        return (2, 1)

class _KMeans(_Base):
    """Choose examples based on k-means cluster centroid distances
    """
    def _G(self, data, K):
        labels, _, _ = Pycluster.kcluster(data.T, K)
        centers, _ = Pycluster.clustercentroids(data.T, clusterid=labels)
        centers = centers.T
        G = zeros((K, data.shape[1]))
        
        for k in range(K):
            D = data - expand_dims(centers[:, k], axis=1)
            G[k, :] = -sqrt(sum(multiply(D, D), axis=0))

        return G
        
class KMX(_KMeans):
    """Choose examples based on k-means cluster centroid distances of X
    """
    def select(self, X, A, S):
        return self._select_per_dictionary(self._G(X, A.shape[1]))

    def group(self):
        return (0, 2)

class KMS(_KMeans):
    """Choose examples based on k-means cluster centroid distances of S
    """
    def select(self, X, A, S):
        return self._select_per_dictionary(self._G(S, A.shape[1]))

    def group(self):
        return (0, 2)

class ErrS(_Base):
    def select(self, X, A, S):
        G = abs(sum(A * S - X, axis=0))
        return self._select_by_sum(G)

    def group(self):
        return (1, 0)

class OLC(_Base):
    """Overlapping cluster selection algorithm
    """
    def select(self, X, A, S):
        Smax = S.max(axis=0)
        
        # Compute the graph G
        P, N = X.shape
        _, K = A.shape
        # Big NxN matrix!
        D = (X.T * X) > .5

        # Guess the number of nonzero in the generated samples
        sp = mean(sum(_is_used(S), axis=0))
        T = N * sp / (10 * K)
        
        # Repeat mk log^2 m times... but just try until we find K clusters
        C = []
        TL = int(ceil(K*sp))
        for _ in range(TL):
            u = randint(N)
            gamma_u = nonzero(asarray(D[u, :]))[1]
            v = gamma_u[randint(len(gamma_u))]
            gamma_uv = asarray(multiply(D[u, :], D[v, :])).squeeze()
            Suv = nonzero(asarray(sum(D[:, gamma_uv], axis = 1) >= T))[0]
            # print "Suv size: %d, current set: %d, target %f" % (len(Suv), len(C), TL)
            
            # Check if a smaller set contains u and v
            smaller_sets = (True for Sab in C if (u in Sab) and (v in Sab) and len(Sab) < len(Suv))

            if len(Suv) > 0 and not(next(smaller_sets, False)):
                # sort Suv by largest S
                # Suv = list(zip(*sorted(zip(Suv,array(S[:, Suv].max(axis=0)).squeeze(axis=0).tolist()), key=itemgetter(1), reverse=True))[0])
                C.append(Suv)
        
        # Now choose examples for each dictionary element (like _select_per_dictionary)
        idx = ones(self.n, dtype = int) * -1
        ki = 0
        for n in range(self.n):
            ki = n % len(C)
            for _ in xrange(K):
                ji = n / len(C)
                if len(C[ki]) > ji:
                    idx[n] = C[ki][ji]
                    break
                ki = (ki + 1) % len(C)
            
            if idx[n] == -1:
                print "Warning: exhausted all C"
                idx[n] = randn(N)
        
        return array(idx)

    def group(self):
        return (2, 0)

#ALL_SELECTORS = [Unif, UsedD, MagS, MagD, MXGS, MXGD, SalMap, SUNS, SUND, KMX, KMS]
ALL_SELECTORS = [Unif, UsedD, MXGS, MXGD, SalMap, SUNS, SUND, KMX, KMS, ErrS, SNRD, SNRS, OLC]

def time_algorithms():
    dictionary=Random(p=10, K=100, sort=False)
    generator = FromDictionaryL1(dictionary, plambda = 1, snr = 6, N = 10000)
    A = dictionary.A
    X, S, _ = generator.generate()
    
    selectors = ALL_SELECTORS
    times = []
    for selector_cls in selectors:
        selector = selector_cls(100)
        trials = []
        for _ in range(10):
            start = time.time()
            selector.select(X, A, S)
            elapsed = time.time() - start
            trials.append(elapsed)
        print "Took %f [s], pm %f [s]" % (mean(array(trials)), std(array(trials)))
        times.append(trials)
    
    return times

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 0:
        import doctest
        doctest.testmod()
    else:
        time_algorithms()
