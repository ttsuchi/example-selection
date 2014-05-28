'''
"True" dictionary elements, used for synthesizing observable examples.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy.random import randn, randint
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from numpy.random import rand

from inc.common import mtr

import matplotlib.cm as cmx
from scipy.io import loadmat

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
    scaling = nanmax(abs(A)) * 2
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

def sort2d(A):
    """Sort the dictionary elements so that, when visualized in a 2D array, similar elements come next to each other.
    
    >>> assert_equal((36, 169), sort2d(randn(36, 169)).shape)
    sorting dictionaries...
    done.    
    """
    print "sorting dictionaries..."
    A = mtr(A.copy())
    K = A.shape[1]

    # the big image size
    Y = int(ceil(sqrt(K)))

    # Create neighbor graph
    neighbors = [[((y-1) % Y)*Y + x, y*Y + ((x-1) % Y), y*Y + ((x+1) % Y), ((y+1) % Y)* Y + x] for x, y in zip(tile(arange(Y), [Y,1]).flatten(), tile(arange(Y), [Y,1]).flatten('F'))]
    neighbors = [[k if k < K else k - Y for k in l] for l in neighbors]

    # Do random swap and try to improve
    for _ in xrange(10000):
        a = randint(K)
        b = randint(K)
        na = neighbors[a]
        nb = neighbors[b]

        E0 = sum(A[:, a].T * A[:, na] + A[:, b].T * A[:, nb])
        E1 = sum(A[:, a].T * A[:, nb] + A[:, b].T * A[:, na])
        
        if E1 > E0:
            A[:, [a, b]] = A[:, [b, a]]

    print "done."
    return A

class DictionarySet(object):
    def __init__(self, A, sort = True, **kwds):
        A = normalize(A)
        if sort:
            A = sort2d(A)
        self.A = A
        self.P, self.K = A.shape

    def coherence(self):
        """Calculate the coherence value for this dictionary, which is the maximum inner product.
        """
        C = dot(self.A.T, self.A)
        return abs(C - diag(diag(C))).max()
    
    def mu(self):
        return self.coherence() * sqrt(self.K)
    
    def mean_coherence(self):
        """Coherency measure
        """
        C = dot(self.A.T, self.A)
        return abs(C - diag(diag(C))).mean()
        

class GeneratedDictionary(DictionarySet):
    def __init__(self, p = 5, K = 50, **kwds):
        self.p = p
        super(GeneratedDictionary, self).__init__(mtr(self.generate_A(p*p, K)), **kwds)

class Random(GeneratedDictionary):
    """Random dictionary

    >>> R=Random(p=8, K=100, sort=False); A=R.A
    
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

    >>> R=RandomGabors(p=8, K=100, sort=False); A=R.A
    
    >>> A.shape
    (64, 100)

    """
    def __init__(self, plambda = -1, psigma = -1, **kwds):
        self.plambda = plambda if plambda > 0 else ceil(kwds['p'] / 2)
        self.psigma = psigma if psigma > 0 else ceil(kwds['p'] / 4)
        super(RandomGabors, self).__init__(**kwds)
    
    def generate_A(self, P, K):
        A=zeros((P, K))

        for ki in range(K):
            F=self.gabor(self.p,self.p,self.plambda,self.psigma)
            A[:,ki]=F.reshape((P, 1))[:,0]

        return A
    
    def gabor(self, width, height, plambda, psigma):
        xv, yv = meshgrid(arange(width), arange(height))
        
        cx = (rand()*0.8 + 0.1)*width
        cy = (rand()*0.8 + 0.1)*height
        lm = (.8 + rand()*.4)*plambda
        sgx = (.8 + rand()*.4)*psigma; sgx = sgx*sgx
        sgy = (.8 + rand()*.4)*psigma; sgy = sgy*sgy
        
        theta = rand() * 2 * pi
        xt = (xv-cx) * cos(theta) + (yv-cy) * sin(theta)
        yt =-(xv-cx) * sin(theta) + (yv-cy) * cos(theta)
        
        return exp(-.5 * ((xt*xt)/sgx+(yt*yt)/sgy)) * cos(2 * pi/lm * xt)

class _FromFile(GeneratedDictionary):
    """Load dictionaries from the specified MATLAB file.
    """
    def __init__(self, **kwds):
        self.D = None
        super(_FromFile, self).__init__(sort = False, **kwds)

    def generate_A(self, P, K):
        A=zeros((P, K))
        
        if self.D is None:
            self.D = self._load_dictionary(P, K)

        N = self.D.shape[1]
        for ki in range(K):
            A[:, ki] = self.D[:, ki % N]

        return A

class Letters(_FromFile):
    """Load dictionaries composed of alphabet letters
    """
    def _load_dictionary(self, P, K):
        return loadmat('../contrib/letters/alphabet-%d.mat' % sqrt(P))['L']
    
class DleRandomGaussian(_FromFile):
    """64x256 Random Gaussian Dictionary from http://www.ux.uis.no/~karlsk/dle/
    """
    def _load_dictionary(self, P, K):
        assert P == 64
        assert K <= 256
        return loadmat('../contrib/dictionaries/dict_rand.mat')['D']

class DleOrthogonal(_FromFile):
    """64x256 orthogonal Dictionary from http://www.ux.uis.no/~karlsk/dle/
    """
    def _load_dictionary(self, P, K):
        assert P == 64
        assert K <= 256
        return loadmat('../contrib/dictionaries/dict_orth.mat')['D']

class DleSine(_FromFile):
    """64x256 separable dictionary with sine elements from http://www.ux.uis.no/~karlsk/dle/
    """
    def _load_dictionary(self, P, K):
        assert P == 64
        assert K <= 256
        return loadmat('../contrib/dictionaries/dict_sine.mat')['D']

def main(plot = False):
    if plot:
        import sys
        # Plot some dictionaries
        import matplotlib.pyplot as plt
        for cls in [Random, RandomGabors, Letters]:            
            A = cls(p=8, K=25).A
            plt.figure()
            plt.imshow(to_image(A), aspect = 'equal', interpolation = 'nearest', vmin = 0, vmax = 1)
            plt.axis('off')
            plt.title(cls.__name__)
            plt.show()
            plt.waitforbuttonpress()
    else:
        import doctest
        doctest.testmod()

if __name__ == '__main__':
    main(len(sys.argv) > 1)

        
            
