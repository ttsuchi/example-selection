'''
Learn a small set of dictionaries from the same dataset used in SPARSENET.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from data.observable import FromImageDataset

from algorithms.selection import Unif, MagS, MagD, MXGS, MXGD
from algorithms.encoding import LASSO, KSparse
from algorithms.updater import GD
from experiment.design import Experiment

def main():
    selectors = [cls(100) for cls in [Unif, MXGD]]
    plambda = 1
    encoders = [KSparse(K=3)]
    updater = GD(encoders[0])
    experiment = Experiment('sparsenet', FromImageDataset('../contrib/sparsenet/IMAGES.mat', p = 16, K = 192),
                            selectors = selectors, encoders = encoders, updaters = [updater])
    experiment.run(400, parallel_jobs = 1, save_every = 5, plot_every = 5)
        
if __name__ == '__main__':
    main()