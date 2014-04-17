'''
Contains function classes for executing the update functions.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import memmap

from joblib import Parallel, delayed
from tempfile import mkdtemp
import os.path as path

class Serial(object):
    """Executes the updates serially.
    """
    def __call__(self, update_fn, X, As, designs, itr):
        return [update_fn(design, X, A, itr) for A, design in zip(As, designs)]

class Parallel(object):
    """Executes the updates in parallel using multiprocessing.    
    """
    def __init__(self, parallel_jobs = 4):
        self.parallel_jobs = parallel_jobs
    
    def __call__(self, update_fn, X, As, designs, itr):
        # TODO this is not working... numpy gets stuck
        X_filename = path.join(mkdtemp(), 'X.dat')
        Xm = memmap(X_filename, dtype = X.dtype, mode='w+', shape = X.shape)
        Xm[:] = X[:]
        return Parallel(n_jobs = self.parallel_jobs)(delayed(update_fn)(design, Xm, A, itr) for A, design in zip(As, designs))
