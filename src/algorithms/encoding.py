'''
The encoding algorithms calculate the activation (S) from the observable (X) and dictionary (A).

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy.random import randn
from spams import lasso, omp, somp

from inc.common import mtr

def equalize_activities(S, eq_power = .5):
    """Equalizes the activity.
    When eq_factor is closer to 1, more equalization takes place
    """
    m = mean(abs(S), axis=1)
    assert m.shape == (S.shape[0], 1)
    
    dead_idx=m < 1e-12
    if any(dead_idx):
        # Fill zero activations with random activations centered around the mean
        if all(dead_idx):
            S = asmatrix(abs(randn(S.shape[0], S.shape[1])))
        else:
            S[nonzero(dead_idx), :] = abs(mean(m) + std(S) * randn(nonzero(dead_idx)[0].size, S.shape[1]))
        m = mean(S, axis=1)

    # Try to equalize variance of mean
    return mtr(multiply(S, power((mean(m) / m) , eq_power)))

def _normalize(A):
    return asfortranarray(A / tile(sqrt(multiply(A,A).sum(axis=0)),(A.shape[0],1)))

class _Base(object):
    def __init__(self, **kwds):
        pass

    def encode(self, X, A):
        """Clean up and equalize the variance.
        """
        S = self._encode(X,_normalize(A))
        S[S<0]=0
        return mtr(S)        

class LASSO(_Base):
    """Solve the LASSO problem using the SPAMS package:

        min_{alpha} 0.5||x-Dalpha||_2^2 + lambda1||alpha||_1 +0.5 lambda2||alpha||_2^2
    
    >>> A = LASSO(plambda = 1).encode(ary(randn(64, 1000)), ary(randn(64, 5)))
    
    """
    def __init__(self, plambda = .15, max_iter = 1000, pos = True, **kwds):
        self.plambda = plambda
        self.iter = max_iter
        self.spams_param = {
            'mode':      2,  
            'lambda1':   plambda,
            'lambda2':   0,
            # 'L':         max_iter,
            'pos':       pos,
            'verbose':   False,
            'numThreads': -1
            }
        super(LASSO, self).__init__(**kwds)
    
    def _encode(self, X, A):
        S = lasso(X, A, return_reg_path = False, **self.spams_param).todense()
        #print "LASSO nnz: %d" % count_nonzero(S)
        return S

class OMP(_Base):
    def __init__(self, nnz = 3, **kwds):
        self.nnz = nnz
        super(OMP, self).__init__(**kwds)
        
    def _encode(self, X, A):
        # Solve min_{alpha} ||x-Dalpha||_2^2  s.t. ||alpha||_0 <= L
        S=omp(X, A, L=self.nnz, return_reg_path = False, numThreads=-1).todense()
        #print "OMP nnz: %d" % count_nonzero(S)
        return S
    
    def __str__(self):
        return "OMP['nnz':%d]" % self.nnz

class SOMP(_Base):
    def __init__(self, nnz = 3, **kwds):
        self.nnz = nnz
        super(SOMP, self).__init__(**kwds)
        
    def _encode(self, X, A):
        # Solve min_{A_i} ||A_i||_{0,infty}  s.t  ||X_i-D A_i||_2^2 <= eps*n_i
        ind_groups = arange(0, X.shape[1], 10, dtype=int32)
        S=somp(X, A, ind_groups, L=self.nnz, numThreads=-1).todense()
        return S

class KSparse(_Base):
    def __init__(self, nnz = 3, **kwds):
        self.nnz = nnz
        super(KSparse, self).__init__(**kwds)

    def _encode(self, X, A):
        """Picks the top K maximum activations for each column.
        
        >>> print KSparse(nnz=2).encode(X=matrix([[1,6,7],[2,5,9],[3,4,8]]), A=eye(3))
        [[ 0.  6.  0.]
         [ 2.  5.  9.]
         [ 3.  0.  8.]]
        """
        S0 = A.T * X
        S = zeros(S0.shape)
        js=arange(S0.shape[1])
        for _ in range(self.nnz):
            idx = argmax(S0, axis=0)
            S[idx,js]=S0[idx,js]
            S0[idx,js]=0
        return S


if __name__ == '__main__':
    import doctest
    doctest.testmod()
