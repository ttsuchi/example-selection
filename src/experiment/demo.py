'''
Experiment 1: simple dictionary learning based on random dictionaries.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from data.observable import FromDictionaryL1
from data.dictionary import Random

from algorithms.selection import Unif, MXGD
from algorithms.encoding import KSparse
from algorithms.learning import GD

from experiment.design import Design

def demo():
    true_dictionary = Random(4, 32)
    selectors = [Unif(100), MXGD(100)]
    encoder = KSparse(K = 2)
    design = Design('demo', FromDictionaryL1(true_dictionary), selectors, encoder, GD(encoder))
    design.run(200, plot = True, plot_every = 10, save_every = 10)

if __name__ == '__main__':
    demo()
