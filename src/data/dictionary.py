'''
"True" dictionary elements, used for synthesizing observable examples.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy import sum
from numpy.random import randn
from numpy.testing import assert_allclose, assert_array_equal

def normalize(A):
    """Normalizes A to have unit norm.
    
    >>> nA = matrix([[0,1,1],[1,0,1]]); A=normalize(nA)

    >>> assert_allclose(sum(multiply(A,A)), 3, atol=1e-10)
    
    >>> assert_allclose(A, matrix([[0,1,sqrt(.5)],[1,0,sqrt(.5)]]))
    """
    return A/(sqrt(sum(multiply(A,A),axis=0))+spacing(1))

class Base(object):
    def __init__(self, p = 5, K = 50, **kwds):
        self.p = p
        self.P = p*p
        self.K = K
        self._A = None
    
    @property
    def A(self):
        if self._A == None:
            self._A = normalize(asmatrix(self.generate_A()))
        return self._A

class Random(Base):
    """Random dictionary

    >>> R=Random(p=8, K=100); A=R.A
    
    >>> A.shape
    (64, 100)
    
    Repeated invocation should return the same:
    
    >>> assert_array_equal(R.A, A)
    
    """
    def generate_A(self):
        return randn(self.P, self.K)

class Lines(Base):
    """Anti-aliased lines
    """
    def generate_A(self):
        sz=array([self.k,self.k])
        ks=linspace(0, pi, self.K+1)
        ks=ks[:self.K]
        A=zeros(prod(sz), self.K)
        
        for ki in range(self.K):
            k=ks[ki]
            m=self.lineMask(sz, self.p*[cos(k),sin(k)]+(sz+1)/2, -self.p*[cos(k),sin(k)]+(sz+1)/2, 1)
            A[:,ki]=m[:]
        
        return A

    def lineMask(self, size, x, y, sigma):
        # TODO write this
        return 0

class RandomGabors(Base):
    """Generate random Gabor patches
    """
    def __init__(self, plambda = -1, sigma = -1, **kwds):
        super(RandomGabors, self).__init__(**kwds)
        self.plambda = plambda if plambda > 0 else ceil(self.p / 2)
        self.sigma = sigma if sigma > 0 else ceil(self.p / 4)
    
    def generate_A(self):
        A=zeros(self.P, self.K)

        for ki in range(self.K):
            F=self.gabor(self.p,self.p,self.plambda,self.psigma)
            A[:,ki]=F[:]
        
        return A
    
    def gabor(self, width, height, plambda, sigma):
        # TODO write gabor
        return 0
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()
