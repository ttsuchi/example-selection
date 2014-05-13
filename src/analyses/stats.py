'''
Calculates various statistics during the learning steps.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy.random import randn
from numpy.testing import assert_allclose, assert_array_equal
import matplotlib.pyplot as plt

from munkres import Munkres

from inc.common import mtr
from data.dictionary import normalize

def collect_stats(X, A, oldA, Astar, S, Sstar, Xsnr, idx):
    """Calculates various tatistics for the given X, A, S
    
    returns (stats, A), where:
        stats: [reconstruction stats across all X, reconstruction stats across currently picked X]
        A: re-ordered dictionary accoring to the best match, if the true dictionary (Astar) is provided
        
    >>> X = matrix([[1, 2, 0, sqrt(.5)], [0, 0, 1, 2+sqrt(.5)]]); A = normalize(matrix([[1,0,1],[0,1,1]])); S = matrix(([1, 2, 0, 0],[0, 0, 1, 2],[0, 0, 0, 1]))
    
    >>> stats, _ = collect_stats(X, A, S, array([0,1])); assert_allclose( stats['stats_all'], 0, atol=1e-10)
    
    >>> _, newA=collect_stats(X, A, S, array([0,1]), normalize(matrix([[1,.9,0],[1.1,0,1]])))
    
    >>> assert_allclose( newA, normalize(matrix([[1,1,0],[1,0,1]])) )
    """
    Xp = X[:,idx]
    R  = X - A*S
    Rp = Xp - A*S[:,idx]
    diff_A = A - oldA 
    Sm = mean(S,axis=1)
    Xc = Xp.T*Xp
    stats = {
        'loss_all':     mean(multiply(R, R)),
        'loss_sampled': mean(multiply(Rp,Rp)),
        'diff_A':       mean(multiply(diff_A, diff_A)),
        'std_S':        std(Sm),
        'mean_S':       mean(Sm),
        'cv':           std(Sm) / mean(Sm),
        'mean_Xp_dist': mean(diag(Xc))-mean(Xc)
    }
    
    if Xsnr is not None:
        Xsnr = asarray(Xsnr).squeeze()
        stats.update({
            'mean_Xsnr':    mean(Xsnr),
            'std_Xsnr':     std(Xsnr),
            'mean_Xsnr_p':  mean(Xsnr[idx]),
            'std_Xsnr_p':   std(Xsnr[idx])
            })
    
    if Astar is None:
        newA = mtr(A.copy())
    else:
        # Calculate distance
        C = - Astar.T * A
        assert all(isfinite(C))
        idx = Munkres().compute(C.tolist())
        newA = mtr(zeros(A.shape))
        newS = zeros(S.shape)
        for r, c in idx:
            newA[:, r] = A[:, c]
            newS[r, :] = S[c, :]

        dA = Astar - newA
        stats['dist_A'] = mean(multiply(dA, dA))
        
        dS = Sstar - newS
        stats['dist_S'] = mean(multiply(dS, dS)) 

    return stats, newA

def _history(stats, column):
    return matrix(map(lambda s: s[column].as_matrix(), stats)).T

def plot_stats(stats, design_names):
    N = 4
    N+= 3 if 'dist_A' in stats[0].columns else 0
    N+= 1 if 'mean_Xsnr' in stats[0].columns else 0
    N+= 1 if 'mean_Xp_dist' in stats[0].columns else 0

    plt.figure(len(design_names) + 1, figsize = (8,6), dpi=80, facecolor='w', edgecolor='k')
    plt.clf()
    
    plt.subplot(N,1,1)
    plt.plot(_history(stats,'loss_sampled'))
    plt.gca().set_xticklabels([])
    plt.title("loss for the sampled set")

    plt.subplot(N,1,2)
    plt.plot(_history(stats,'loss_all'))
    plt.gca().set_xticklabels([])
    plt.title("loss for all training set")

    plt.subplot(N,1,3)
    plt.plot(_history(stats,'diff_A'))
    plt.title("difference in A")

    if 'dist_A' in stats[0].columns:
        plt.subplot(N,1,4)
        plt.plot(_history(stats,'dist_A'))
        plt.gca().set_xticklabels([])
        plt.title("average distance from the true dictionary")

        plt.subplot(N,1,5)
        data = _history(stats,'dist_A')[-1,:].T
        ind = arange(data.shape[0])
        width = .8
        plt.bar(ind, data, width, color = 'b')
        plt.gca().set_xticks(ind+width/2)
        plt.gca().set_xticklabels(design_names)
        
        plt.subplot(N,1,6)
        plt.plot(_history(stats,'dist_S'))
        plt.title("average distance from the true activation")

    if 'mean_Xsnr' in stats[0].columns:
        plt.subplot(N,1,7)
        plt.plot(_history(stats,'mean_Xsnr_p'))
        plt.title("SNR of selected examples")
        
    if 'mean_Xp_dist' in stats[0].columns:
        plt.subplot(N,1,8)
        plt.plot(_history(stats,'mean_Xp_dist'))
        plt.title("Mean distances of selected examples")

    plt.subplot(N,1,N)
    plt.plot(zeros((2,len(design_names))))
    plt.axis('off')
    plt.legend(design_names, loc='center', ncol=4)    

if __name__ == '__main__':
    import doctest
    doctest.testmod()