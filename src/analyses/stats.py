'''
Calculates various statistics during the learning steps.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy.random import randn
from numpy.testing import assert_allclose, assert_array_equal
from scipy.cluster.vq import kmeans, vq, whiten
import matplotlib.pyplot as plt

from munkres import Munkres

from inc.common import mtr
from data.dictionary import normalize

def collect_generator_stats(self, A_signal, noise, itr):
    """ Mixin method for the generator class. Calculates some stats related to the trainging examples.
    """
    # Calculate the SNR for each example
    A_noise  = sqrt(mean(multiply(noise, noise), axis=0))
    self.Xsnr = 20 * (log10(A_signal / A_noise))

    if (itr % 10 == 1):
        # Perform k-means
        k = 10
        W = whiten(self.X.T)
        self.codebook, _ = kmeans(W, k)

def collect_stats(generator, S, oldA, A, idx, itr):
    """Calculates various tatistics for the given X, A, S
    
    returns (stats, A), where:
        stats: [reconstruction stats across all X, reconstruction stats across currently picked X]
        A: re-ordered dictionary accoring to the best match, if the true dictionary (Astar) is provided
        
    >>> X = matrix([[1, 2, 0, sqrt(.5)], [0, 0, 1, 2+sqrt(.5)]]); A = normalize(matrix([[1,0,1],[0,1,1]])); S = matrix(([1, 2, 0, 0],[0, 0, 1, 2],[0, 0, 0, 1]))
    
    >>> stats, _ = collect_stats(X, A, S, array([0,1])); assert_allclose( stats['stats_all'], 0, atol=1e-10)
    
    >>> _, newA=collect_stats(X, A, S, array([0,1]), normalize(matrix([[1,.9,0],[1.1,0,1]])))
    
    >>> assert_allclose( newA, normalize(matrix([[1,1,0],[1,0,1]])) )
    """
    X = generator.X
    Xp = X[:,idx]
    R  = X - A*S
    Rp = Xp - A*S[:,idx]
    diff_A = A - oldA 
    Sm = mean(S,axis=1)
    Xc = Xp[:,:1000].T*Xp[:,:1000]
    stats = {
        'loss_all':     mean(multiply(R, R)),
        'loss_sampled': mean(multiply(Rp,Rp)),
        'diff_A':       mean(multiply(diff_A, diff_A)),
        'std_S':        std(Sm),
        'mean_S':       mean(Sm),
        'cv':           std(Sm) / mean(Sm),
        'mean_Xp_dist': mean(diag(Xc))-mean(Xc),
        'vqd':          _vqd(generator, idx)
    }
    
    if hasattr(generator, 'Xsnr'):
        Xsnr = asarray(generator.Xsnr).squeeze()
        stats.update({
            'mean_Xsnr':    mean(Xsnr),
            'std_Xsnr':     std(Xsnr),
            'mean_Xsnr_p':  mean(Xsnr[idx]),
            'std_Xsnr_p':   std(Xsnr[idx])
            })
    
    if hasattr(generator, 'dictionary'):
        # Calculate distance
        Astar = generator.dictionary.A
        newA, newS = _best_match(Astar, A, S, itr)

        dA = Astar - newA
        stats['dist_A'] = mean(multiply(dA, dA))
        
        dS = generator.S - newS
        stats['dist_S'] = mean(multiply(dS, dS)) 
    else:
        newA = mtr(A.copy())

    return stats, newA

def _best_match(Astar, A, S, itr):
    """Calculates the best matching ordering for A against Astar.
       If there are many dictionaries, Munkres can take a bit too long.
       So the matching is only done at logarithmically spaced epochs, [1,2,3,4,5,6,7,8,10,12,14,17,20,24,...]
    """
    q = 15
    if floor(q*log10(itr+1)) != floor(q*log10(itr+2)):
        C = - Astar.T * A
        assert all(isfinite(C))
        idx = Munkres().compute(C.tolist())
        newA = mtr(zeros(A.shape))
        newS = zeros(S.shape)
        for r, c in idx:
            newA[:, r] = A[:, c]
            newS[r, :] = S[c, :]
        
        return newA, newS
    else:
        return A, S

def _vqd(generator, idx, k = 10):
    """Calculate the vector-quantized histogram intersection distance.
       The codebook is derived from the k-means clustering of X, and the histograms of X and Xp are compared
       using the histogram interesection distance.
       
       >>> X = matrix(randn(64, 10000)); idx = arange(100); d = _vqd(X, idx, k = 10)
       
       >>> assert(d > 0); assert(d < 0.2);  # The distance should be fairly small
    """
    if hasattr(generator, 'codebook'):
        W = whiten(generator.X.T)
        hX = _hist(W, generator.codebook)
        hXn= _hist(W[idx, :], generator.codebook)
        d = 1 - sum(array([hXn, hX]).min(axis=0))
        return d
    else:
        return NaN

def _hist(X, codebook):
    code, _ = vq(X, codebook)
    h, _ = histogram(code, bins=codebook.shape[0]-1)
    h = array(h, dtype=float)
    return h / sum(h)
    

def _history(stats, column):
    return matrix(map(lambda s: s[column].as_matrix(), stats)).T

def plot_stats(stats, design_names):
    N = 1
    N+= 3 if 'dist_A' in stats[0].columns else 0
#     N+= 1 if 'mean_Xsnr' in stats[0].columns else 0
#     N+= 1 if 'mean_Xp_dist' in stats[0].columns else 0
    N+= 1 if 'vqd' in stats[0].columns else 0
    n = 1

    plt.figure(len(design_names) + 1, figsize = (8,6), dpi=80, facecolor='w', edgecolor='k')
    plt.clf()
    
#     plt.subplot(N,1,n); n += 1
#     plt.plot(_history(stats,'loss_sampled'))
#     plt.gca().set_xticklabels([])
#     plt.title("loss for the sampled set")
# 
#     plt.subplot(N,1,n); n += 1
#     plt.plot(_history(stats,'loss_all'))
#     plt.gca().set_xticklabels([])
#     plt.title("loss for all training set")

    plt.subplot(N,1,n); n += 1
    plt.plot(_history(stats,'diff_A'))
    plt.gca().set_yscale('log')
    plt.title("difference in A")

    if 'dist_A' in stats[0].columns:
        plt.subplot(N,1,n); n += 1
        plt.plot(_history(stats,'dist_A'))
        plt.gca().set_xticklabels([])
        plt.title("average distance from the true dictionary")

#         plt.subplot(N,1,n); n += 1
#         data = _history(stats,'dist_A')[-1,:].T
#         ind = arange(data.shape[0])
#         width = .8
#         plt.bar(ind, data, width, color = 'b')
#         plt.gca().set_xticks(ind+width/2)
#         plt.gca().set_xticklabels(design_names)
#         
#         plt.subplot(N,1,n); n += 1
#         plt.plot(_history(stats,'dist_S'))
#         plt.title("average distance from the true activation")

#     if 'mean_Xsnr' in stats[0].columns:
#         plt.subplot(N,1,n); n += 1
#         plt.plot(_history(stats,'mean_Xsnr_p'))
#         plt.title("SNR of selected examples")
#         
#     if 'mean_Xp_dist' in stats[0].columns:
#         plt.subplot(N,1,n); n += 1
#         plt.plot(_history(stats,'mean_Xp_dist'))
#         plt.title("Mean distances of selected examples")

    if 'vqd' in stats[0].columns:
        plt.subplot(N,1,n); n += 1
        data = _history(stats,'vqd')
        plt.plot(data)
#         print data.shape
#         plt.plot(arange(size(data))[isfinite(data)], data[isfinite(data)])
        plt.title("VQd")

    plt.subplot(N,1,N)
    plt.plot(zeros((2,len(design_names))))
    plt.axis('off')
    plt.legend(design_names, loc='center', ncol=4)    

if __name__ == '__main__':
    import doctest
    doctest.testmod()