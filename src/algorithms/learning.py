'''
Represents a single step of dictionary learning iteration.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy.random import randn
from numpy.testing import assert_allclose, assert_array_equal

from munkres import Munkres

from data.dictionary import normalize

def evaluate_loss(X, A, S, idx, Astar = None):
    """Evaluates the loss values for the given X, A, S
    
    returns (loss, A), where:
        loss: [reconstruction loss across all X, reconstruction loss across currently picked X]
        A: re-ordered dictionary accoring to the best match, if the true dictionary (Astar) is provided
        
    >>> X = matrix([[1, 2, 0, sqrt(.5)], [0, 0, 1, 2+sqrt(.5)]]); A = normalize(matrix([[1,0,1],[0,1,1]])); S = matrix(([1, 2, 0, 0],[0, 0, 1, 2],[0, 0, 0, 1]))
    
    >>> loss, _ = evaluate_loss(X, A, S, array([0,1])); assert_allclose( loss['loss_all'], 0, atol=1e-10)
    
    >>> _, newA=evaluate_loss(X, A, S, array([0,1]), normalize(matrix([[1,.9,0],[1.1,0,1]])))
    
    >>> assert_allclose( newA, normalize(matrix([[1,1,0],[1,0,1]])) )
    """
    R = X - A*S; Rp = X[:,idx] - A*S[:,idx]
    loss = {
        'loss_all': mean(multiply(R, R)),
        'loss_sampled': mean(multiply(Rp,Rp))
    }
    
    newA = None
    if Astar is not None:
        # Calculate conformity
        C = 1 - abs(Astar.T * A)
        idx = Munkres().compute(C.tolist())
        newA = asmatrix(zeros(A.shape))
        for r, c in idx:
            newA[:, r] = A[:, c]

        loss['conformity'] = mean(abs(Astar.T * newA))

    return loss, newA

def update_with(design, X, A, selector):
    """Return a new dictionary using the examples picked by the current selection policy.
    """
    stats = {}
    
    # Encode all training examples
    S = design.encoder.encode(X, A)
    
    # Pick examples to learn from
    idx = selector.select(X, A, S)
    
    # Update dictionary using these examples
    newA = design.updater.update(X[:, idx], A)
    delta = newA - A
    stats['std'] = sqrt(mean(multiply(delta, delta)))
    
    # Evaluate the loss (and reorder A, if necessary)
    loss, A = evaluate_loss(X, newA, S, idx, design.Astar)
    stats.update(loss)

    # Some top chosen examples
    Xp = X[:, idx[:min(len(idx), A.shape[1])]]
    return A, stats, Xp

if __name__ == '__main__':
    import doctest
    doctest.testmod()
