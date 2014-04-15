'''
Experiment 1: learn Gabor filter-generated samples.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from data.observable import FromDictionaryL1
from data.dictionary import RandomGabors

from algorithms.selection import Unif, MagS, MagD, MXGS, MXGD
from algorithms.encoding import KSparse
from algorithms.updater import GD
from experiment.design import Design

def main():
    true_dictionary = RandomGabors(p = 6, K = 13*13)
    selectors = [cls(100) for cls in [Unif, MagS, MagD, MXGS, MXGD]]
    encoder = KSparse(K = 3)
    design = Design('demo', FromDictionaryL1(true_dictionary, N = 5000), selectors, encoder, GD(encoder))
    design.run(200, parallel_jobs = 1, save_every = 5, plot_every = 5)
        
if __name__ == '__main__':
    main()
