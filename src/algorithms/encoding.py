'''
The encoding algorithms calculate the activation (S) from the observable (X) and dictionary (A).

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from spams import lasso, somp

class Base(object):
    def __init__(self, **kwds):
        pass

class LASSO(Base):
    def __init__(self, plambda, max_iter = 1000, **kwds):
        self.plambda = plambda
        self.iter = max_iter
        self.spams_param = {
            # Solve min_{alpha} 0.5||x-Dalpha||_2^2 + lambda1||alpha||_1 +0.5 lambda2||alpha||_2^2
            'mode':      2,  
            'lambda1':   plambda,
            'lambda2':   0,
            'L':         max_iter,
            'pos':       True,
            'numThreads': -1
            }
        super(LASSO, self).__init__(**kwds)
    
    def encode(self, X, A):
        S = lasso(X, A, return_reg_path = False)
        S[S<0]=0
        return asmatrix(S)

class SOMP(Base):
    def __init__(self, K = 3, **kwds):
        self.K = K
        super(SOMP, self).__init__(**kwds)
        
    def encode(self, X, A):
        # Solve min_{A_i} ||A_i||_{0,infty}  s.t  ||X_i-D A_i||_2^2 <= eps*n_i
        ind_groups = array(xrange(0, X.shape[1], 10), dtype=int64)
        S=somp(X, A, ind_groups,L=self.K,numThreads=-1)
        return asmatrix(S)

class KSparse(Base):
    def __init__(self, K = 3, **kwds):
        self.K = K
        super(KSparse, self).__init__(**kwds)

    def encode(self, X, A):
        """Picks the top K maximum activations for each column.
        
        >>> print KSparse(K=2).encode(X=matrix([[1,6,7],[2,5,9],[3,4,8]]), A=eye(3))
        [[ 0.  6.  0.]
         [ 2.  5.  9.]
         [ 3.  0.  8.]]
        """
        S0 = A.T * X
        S = zeros(S0.shape)
        js=arange(S0.shape[1])
        for _ in range(self.K):
            idx = argmax(S0, axis=0)
            S[idx,js]=S0[idx,js]
            S0[idx,js]=0
        return asmatrix(S)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
