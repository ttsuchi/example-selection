'''
Demo experiment: simple dictionary learning based on random dictionaries.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from data.observable import FromDictionaryL1
from data.dictionary import Random

from algorithms.selection import Unif, MXGD
from algorithms.encoding import KSparse
from algorithms.updater import GD
from experiment.design import Experiment

def main():
    true_dictionary = Random(4, 32)
    selectors = [cls(100) for cls in [Unif, MXGD]]
    encoder = KSparse(K = 2)
    updater = GD(encoder)
    experiment = Experiment('demo', FromDictionaryL1(true_dictionary),
                            selectors = selectors, encoders = [encoder], updaters = [updater])
    experiment.run(200, parallel_jobs = 1, save_every = 10, plot_every = 5)
        
if __name__ == '__main__':
    main()