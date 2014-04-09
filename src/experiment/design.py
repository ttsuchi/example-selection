'''
Executes a dictionary learning experiment.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy.random import randn
from numpy.testing import assert_allclose, assert_array_equal

from scipy.io import savemat
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

from munkres import Munkres

from data.observable import FromDictionaryL1
from data.dictionary import Random, normalize

import datetime
import time

from joblib import Parallel, delayed
from tempfile import mkdtemp
import os.path as path

def evaluate_loss(X, A, S, idx, Astar = None):
    """Evaluates the loss values for the given X, A, S
    
    returns (loss, A), where:
        loss: [reconstruction loss across all X, reconstruction loss across currently picked X]
        A: re-ordered dictionary accoring to the best match, if the true dictionary (Astar) is provided
        
    >>> X = matrix([[1, 2, 0, sqrt(.5)], [0, 0, 1, 2+sqrt(.5)]]); A = normalize(matrix([[1,0,1],[0,1,1]])); S = matrix(([1, 2, 0, 0],[0, 0, 1, 2],[0, 0, 0, 1]))
    
    >>> assert_allclose( evaluate_loss(X, A, S, array([0,1])), array([0,0]), atol=1e-10)
    
    >>> c, newA=evaluate_loss(X, A, S, array([0,1]), normalize(matrix([[1,.9,0],[1.1,0,1]])))
    
    >>> assert_allclose( newA, normalize(matrix([[1,1,0],[1,0,1]])) )
    """
    R = X - A*S; Rp = X[:,idx] - A*S[:,idx]
    loss = [
        mean(multiply(R, R)),
        mean(multiply(Rp,Rp))
        ]
    
    if Astar is not None:
        C = 1 - abs(Astar.T * A)
        idx = Munkres().compute(C.tolist())
        newA = asmatrix(zeros(A.shape))
        for r, c in idx:
            newA[:, r] = A[:, c]
        
        # conformity
        loss.append(mean(abs(Astar.T * newA)))
        return array(loss), newA
    else:
        return array(loss)

def update_with_selection(design, X, A, selector):
    """Return a new dictionary using the examples picked by the current selection policy.
    """
    S = design.encoder.encode(X, A)
    
    # Pick examples to learn from
    idx = selector.select(X, A, S)
    
    # Update dictionary using these examples
    A = design.updater.update(X[:, idx], A)
    
    # Evaluate the loss (and reorder A, if necessary)
    
    losses, A = evaluate_loss(X, A, S, idx, design.Astar)
    
    return A, losses


class Design(object):
    """Create and execute a dictionary learning experiment.
    """
    
    SAVE_DIR='../results/'
    
    def __init__(self, name, observables, selectors, encoder, updater, **kwds):
        self.name = name
        self.observables = observables
        self.Astar =  self.observables.dictionary.A if hasattr(self.observables, 'dictionary') else None
        self.selectors = selectors
        self.selector_names = map(lambda selector: selector.name, selectors)
        self.encoder = encoder
        self.updater = updater
        self.X_filename = path.join(mkdtemp(), 'X.dat')


    def to_dict(self):
        return {
            'observables': self.observables.__dict__,
            'selectors'  : [selector.__dict__ for selector in self.selectors],
            'encoder'    : self.encoder.__dict__,
            'updater'    : self.updater.__dict__
            }


    def update(self, num_iter, parallel_jobs):
        """Generator that performs the dictionary update step.
        """
        # Initial dictionary set
        A = Random(self.observables.p, self.observables.K).A
        
        # List to store current state
        results = [(A.copy(), None) for _ in self.selectors]

        for itr in range(num_iter):
            start = time.time()

            # Generate mini-batches
            X = self.observables.sample()
            if parallel_jobs > 1:
                Xm = memmap(self.X_filename, dtype = X.dtype, mode='w+', shape = X.shape)
                Xm[:] = X[:]
                results = Parallel(n_jobs = parallel_jobs)(delayed(update_with_selection)(self, Xm, results[i][0], selector) for i, selector in enumerate(self.selectors))
            else:
                results = [update_with_selection(self, X, results[i][0], selector) for i, selector in enumerate(self.selectors)]
            
            elapsed = (time.time() - start)
            
            yield results, elapsed, itr
    
    def run(self, num_iter = 200, plot = False, plot_every = 10, save_every = 10, parallel_jobs = 1):
        all_results = []
        
        for results, elapsed, itr in self.update(num_iter, parallel_jobs):
            all_results.append((results, elapsed))
                
            if mod(itr, save_every) == 1:
                self.save(all_results)

            if plot and mod(itr, plot_every) == 1:
                self.plot(all_results)

            estimated_finish = (datetime.datetime.now() + datetime.timedelta(0, elapsed*(num_iter - itr - 1)))
            print("iter=%3d / %3d, %f[s] elapsed, estimated finish at %s\n" % 
                  (itr+1,num_iter,elapsed,estimated_finish.strftime('%x %X')))

        self.save(all_results)

        if plot:
            self.plot(all_results)
    
    def save(self, all_results):
        savemat(Design.SAVE_DIR + datetime.datetime.now().strftime("%Y%m%d") + self.name,
                {'all_results': all_results, 'parameters': self.to_dict()}, oned_as = 'column')

    def plot(self, all_results):
        all_losses     = matrix([[loss[0] for _, loss in result[0]] for result in all_results])
        sampled_losses = matrix([[loss[1] for _, loss in result[0]] for result in all_results])        
        conformity = matrix([[loss[2] for _, loss in result[0]] for result in all_results]) if self.Astar is not None else None
        
        plt.figure(1)
        plt.clf()
        self.plot_losses(all_losses, sampled_losses, conformity)
        plt.draw()
        
        # last element
        last_result = all_results[-1]
        As = [A for A, _ in last_result[0]]

        for p, selector in enumerate(self.selectors):
            plt.figure(2 + p)
            plt.clf()
            self.plot_current_A(selector, As[p])
            plt.draw()
        
        plt.pause(1)
    
    def plot_losses(self, all_losses, sampled_losses, conformity = None):
        N = 2 if conformity is None else 3
        
        plt.subplot(N,1,1)
        plt.plot(sampled_losses) 
        plt.legend(self.selector_names)
        plt.title("loss for the sampled set")

        plt.subplot(N,1,2)
        plt.plot(all_losses)
        plt.legend(self.selector_names)
        plt.title("loss for all training set")

        plt.subplot(N,1,3)
        plt.plot(conformity)
        plt.legend(self.selector_names)
        plt.title("Average conformity")

    def plot_current_A(self, selector, A):
        rows = 1 if self.Astar is None else 2
        K = min(8, self.observables.K)
        self.subplot_A(A, K, rows, 0)
        plt.title("Current dictionaries for " + selector.name)
        
        if self.Astar is not None:
            self.subplot_A(self.Astar, K, rows, 1)
            plt.title("True dictionaries")            
    
    def subplot_A(self, A, K, rows, row):
        scaling = max(abs(nanmin(A)), nanmax(A))
        for k in range(K):
            a = A[:, k].reshape((self.observables.p, self.observables.p))
            plt.subplot(rows, K, row * K + k)
            plt.imshow(a, cmap = plt.get_cmap('RdBu'), aspect = 'equal', interpolation = 'none', vmin = -scaling, vmax=scaling)
            plt.axis('off')           
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()
