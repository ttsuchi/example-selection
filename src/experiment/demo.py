'''
Experiment 1: simple dictionary learning based on random dictionaries.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from data.observable import FromDictionaryL1
from data.dictionary import Random

import matplotlib.pyplot as plt

from algorithms.selection import Unif, MXGD
from algorithms.encoding import KSparse
from algorithms.updater import GD
from experiment.design import Design

def demo():
    true_dictionary = Random(4, 32)
    selectors = [Unif(100), MXGD(100)]
    encoder = KSparse(K = 2)
    design = Design('demo', FromDictionaryL1(true_dictionary), selectors, encoder, GD(encoder))
    design.run(200, parallel_jobs = 1, save_every = 10, plot_every = 5)
    
    print "Done, press a key on the figure"
    plt.waitforbuttonpress()


        
if __name__ == '__main__':
    demo()
