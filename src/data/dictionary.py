'''
"True" dictionary elements, used for synthesizing observable examples.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy import sum
from numpy.random import randn
from numpy.testing import assert_allclose, assert_array_equal

import matplotlib.cm as cmx

def normalize(A):
    """Normalizes A to have unit norm.
    
    >>> nA = matrix([[0,1,1],[1,0,1]]); A=normalize(nA)

    >>> assert_allclose(sum(multiply(A,A)), 3, atol=1e-10)
    
    >>> assert_allclose(A, matrix([[0,1,sqrt(.5)],[1,0,sqrt(.5)]]))
    """
    return A/(sqrt(sum(multiply(A,A),axis=0))+spacing(1))

def to_image(A, border = 1, colorize = True):
    """Creates a big image for the whole dictinoary set.
    
    >>> A = normalize(matrix([[2,0,0],[0,-2,0],[0,0,2],[0,0,0]]))
    
    >>> assert_array_equal(to_image(A, colorize = False), array([[1,.5,.5,.5,0],[.5,.5,.5,.5,.5],[.5,.5,.5,.5,.5],[.5,.5,.5,.5,.5],[1,.5,.5,.5,.5]]))

    >>> assert_allclose(to_image(A, colorize = True)[2,2], array([0.9657055,   0.96724337,  0.9680892,   1.]))
    """
    A = A.copy()  # It's passed by reference!
    P, K = A.shape
    
    # image patch size
    sz = (int(ceil(sqrt(P))), int(ceil(sqrt(P))))
    
    # the big image size
    SZ = (int(ceil(sqrt(K))), int(ceil(sqrt(K))))
    
    image = ones(( (border+sz[0])*SZ[0]-border, (border+sz[1])*SZ[1]-border )) * .5
    
    # scale images but with 0.5 as the center
    scaling = max(abs(nanmin(A)), nanmax(A)) * 2
    A /= scaling # A in [-.5,.5]
    A += .5      # A in [0, 1]
    
    for k in range(K):
        x = (k % SZ[1]) * (border + sz[1])
        y = (k / SZ[1]) * (border + sz[0])
        image[y:(y+sz[0]),x:(x+sz[1])] = A[:, k].reshape(sz)
    
    if colorize:
        smap = cmx.ScalarMappable(cmap=cmx.get_cmap('RdBu'))
        smap.set_clim(0, 1.0)
        image = smap.to_rgba(image)
        
    return image

class DictionarySet(object):
    def __init__(self, A, **kwds):
        self.A = normalize(A)
        self.P, self.K = A.shape

class GeneratedDictionary(DictionarySet):
    def __init__(self, p = 5, K = 50, **kwds):
        self.p = p
        super(GeneratedDictionary, self).__init__(asmatrix(self.generate_A(p*p, K)), **kwds)

class Random(GeneratedDictionary):
    """Random dictionary

    >>> R=Random(p=8, K=100); A=R.A
    
    >>> A.shape
    (64, 100)
    
    Repeated invocation should return the same:
    
    >>> assert_array_equal(R.A, A)
    
    """
    def generate_A(self, P, K):
        return randn(P, K)

class Lines(GeneratedDictionary):
    """Anti-aliased lines
    """
    def generate_A(self, P, K):
        sz=array([self.k,self.k])
        ks=linspace(0, pi, self.K+1)
        ks=ks[:self.K]
        A=zeros((prod(sz), self.K))
        
        for ki in range(self.K):
            k=ks[ki]
            m=self.lineMask(sz, self.p*[cos(k),sin(k)]+(sz+1)/2, -self.p*[cos(k),sin(k)]+(sz+1)/2, 1)
            A[:,ki]=m[:]
        
        return A

    def lineMask(self, size, x, y, sigma):
        # TODO write this
        return 0

class RandomGabors(GeneratedDictionary):
    """Generate random Gabor patches
    """
    def __init__(self, plambda = -1, sigma = -1, **kwds):
        self.plambda = plambda if plambda > 0 else ceil(self.p / 2)
        self.sigma = sigma if sigma > 0 else ceil(self.p / 4)
        super(RandomGabors, self).__init__(**kwds)
    
    def generate_A(self, P, K):
        A=zeros((P, K))

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
