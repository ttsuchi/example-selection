'''
Defines the common datatype declaration.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *

def ary(X):
    """Returns the array we use for the project. Fortran order is needed for the SPAMS package.
    """
    return asarray(X, dtype=float64, order='F')

def mtr(X):
    return asmatrix(ary(X))