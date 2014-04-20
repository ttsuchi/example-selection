'''
Contains function classes for executing the update functions.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import memmap

from IPython.parallel import Client

from inc.common import mtr

from tempfile import mkdtemp
import os.path as path

class Serial(object):
    """Executes the updates serially.
    """
    def __call__(self, update_fn, X, As, designs, itr):
        return [update_fn(design, X, A, itr) for A, design in zip(As, designs)]

class IParallel(object):
    """Executes the updates in parallel using IPython.parallel.    
    """
    def __init__(self):
        self.client = Client()
        self.dview = self.client[:]
        
    def __call__(self, update_fn, X, As, designs, itr):
        X_filename = path.join(mkdtemp(), 'X.dat')
        Xm = memmap(X_filename, dtype = X.dtype, mode='w+', shape = X.shape)
        Xm[:] = X[:]
        return self.dview.map_sync(UpdateFunction(update_fn, mtr(Xm), itr) , zip(As, designs))

class UpdateFunction(object):
    """A picklable function object for the update function.
    """
    def __init__(self, update_fn, X, itr):
        self.update_fn = update_fn
        self.X = X
        self.itr = itr
    
    def __call__(self, A_design):
        A, design = A_design
        return self.update_fn(design, self.X, A, self.itr)
