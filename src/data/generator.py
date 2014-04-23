'''
Generates observable signals (X).

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy.linalg import eigh
from numpy.random import randn, rand, randint
from numpy.testing import assert_allclose, assert_equal, assert_array_less

from scipy.sparse import csc_matrix
from scipy.io import loadmat

from dictionary import Random

from inc.common import mtr

def whiten(X):
    """Whiten the signal X so that it has zero mean and zero cross-correlation.

    From http://xcorr.net/2011/05/27/whiten-a-matrix-matlab-code/.

    >>> P=10; W=whiten(asmatrix(randn(P, 100)))
    
    >>> assert_allclose(mean(W,axis=1), zeros((P,1)), atol=1e-10)
    
    >>> assert_allclose(var(W,axis=1),  ones((P,1)), atol=1e-10)
    
    >>> assert_allclose(cov(W)-diag(diag(cov(W))), zeros((P,P)), atol=1e-10)
    """
    X = asmatrix(X - mean(asmatrix(X),axis=1))
    C = X * X.T / X.shape[1]
    d, V = eigh(C)
    d[d<0] = 0 # In case d returns very small negative eigenvalues
    return (V / sqrt(d+spacing(1))) * V.T * X

def snr_to_sigma(snr):
    """Convert noise SNR value to STD for noise generation, assuming the signal has unit variance
        (which the whiten() method does.)
        
    >>> assert_allclose(var(snr_to_sigma(.3)*randn(10000,1)), 1e-3, rtol=1e-1) # .3 dB
    
    """
    return 10**(-snr*5)

class Base(object):
    def __init__(self, p = 12, K = 256, N = 10000, **kwds):
        self.p = p # patch dimensions
        self.P = p*p
        self.K = K
        self.N = N # Number of samples to generate on each call

class FromDictionary(Base):
    """Generate examples from a set of "ground-truth" dictionary elements
    
    >>> Astar = Random(8, 100, sort=False); Xgen = FromDictionary(Astar)
    
    >>> assert_equal(Xgen.p, 8)
    
    >>> assert_equal(Xgen.K, 100)
    
    """
    def __init__(self, dictionary, snr = 6, **kwds):
        self.dictionary = dictionary
        self.snr        = snr        # Signal-to-noise ratio in dB
        super(FromDictionary, self).__init__(p = dictionary.p, K = dictionary.K, **kwds)
    
    def generate(self):
        S = self.generate_S()
        X = self.dictionary.A*S
        return mtr(X + randn(X.shape[0], X.shape[1])*snr_to_sigma(self.snr))

class FromDictionaryL0(FromDictionary):
    """Generate examples from a set of "ground-truth" dictionary elements, using L0 sparsity

    >>> Astar = Random(4, 100, sort=False); Xgen = FromDictionaryL0(Astar, nnz = 2)
    
    >>> assert_equal(Xgen.p, 4); assert_equal(Xgen.K, 100)
    
    >>> assert_equal(sum(Xgen.generate_S()), 2*Xgen.N)

    """
    def __init__(self, dictionary, nnz = 3, lambdaS = 10, **kwds):
        self.nnz = nnz  # Sparsity, number of nonzeros
        self.lambdaS = lambdaS # Maginitude sparsity
        super(FromDictionaryL0, self).__init__(dictionary, **kwds)
    
    def generate_S(self):
        rows = randint(self.dictionary.K, size=self.N*self.nnz)
        cols = arange(self.N).repeat(self.nnz)
        data = -log(rand(self.N*self.nnz)) / self.lambdaS + 1
        return mtr(csc_matrix((data, (rows, cols)), shape=(self.dictionary.K, self.N)).todense())

class FromDictionaryL1(FromDictionary):
    """Generate examples from a set of "ground-truth" dictionary elements, using L1 sparsity

    >>> Astar = Random(4, 100, sort=False); Xgen = FromDictionaryL1(Astar, lambdaS = 0.5, N = 500)
    
    >>> assert_equal(Xgen.p, 4); assert_equal(Xgen.K, 100); assert_equal(Xgen.lambdaS, 0.5); assert_equal(Xgen.N, 500)
    
    >>> assert_equal(Xgen.generate().shape, (Xgen.P, Xgen.N))
    
    >>> assert_array_less(-Xgen.generate_S(), zeros((100, 500))+spacing(1)) # Make sure all are non-negative

    """
    def __init__(self, dictionary, lambdaS = 10, **kwds):
        self.lambdaS = lambdaS  # Sparsity
        super(FromDictionaryL1, self).__init__(dictionary, **kwds)
    
    def generate_S(self):
        return mtr(-log(rand(self.dictionary.K, self.N)) / self.lambdaS)

class FromImageDataset(Base):
    """Generate samples from the IMAGES.mat file from http://redwood.berkeley.edu/bruno/sparsenet/.
    
    >>> Xgen = FromImageDataset('../../contrib/sparsenet/IMAGES.mat', p = 16, K=192)
    
    >>> assert_equal(Xgen.generate().shape, (256, Xgen.N))
    
    >>> print isfortran(Xgen.generate())
    True
    
    """
    
    def __init__(self, images_mat, **kwds):
        super(FromImageDataset, self).__init__(**kwds)
        self.images = loadmat(images_mat)['IMAGES']

    def generate(self):
        image_size, _, num_images = self.images.shape
        this_image = self.images[:, :, randint(num_images)].squeeze()
        BUFF = 4
        
        X = mtr(zeros((self.P, self.N)))
        for n in range(self.N):
            r=BUFF+randint(image_size-self.p-2*BUFF)
            c=BUFF+randint(image_size-self.p-2*BUFF)
            X[:,n]=this_image[r:(r+self.p), c:(c+self.p)].reshape([self.P, 1])
        return X


if __name__ == '__main__':
    import doctest
    doctest.testmod()
