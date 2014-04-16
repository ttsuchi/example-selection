'''
Small experiment: learn a small set of random dictionaries using all selection policies.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from data.observable import FromDictionaryL1
from data.dictionary import RandomGabors

from algorithms.selection import Unif, MagS, MagD, MXGS, MXGD
from algorithms.encoding import LASSO, KSparse
from algorithms.updater import GD
from experiment.design import Experiment

def main():
    true_dictionary = RandomGabors(p = 6, K = 36)
    selectors = [cls(100) for cls in [Unif, MagS, MagD, MXGS, MXGD]]
    plambda = 1
    encoders = [LASSO(plambda, K = 2), KSparse(K = 2)]
    updater = GD(encoders[0])
    experiment = Experiment('small', FromDictionaryL1(true_dictionary, lambdaS = plambda),
                            selectors = selectors, encoders = encoders, updaters = [updater])
    experiment.run(400, parallel_jobs = 1, save_every = 5, plot_every = 5)
        
if __name__ == '__main__':
    main()