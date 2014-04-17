#! /usr/bin/env python
'''
Loads a pickled file (.pkl) from the ../results/ directory and plots them.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
import sys
import os
from experiment.design import Experiment

import matplotlib.pyplot as plt

def main(name):
    state = State.load(name)
    state.plot()

    print "Done, close the figures to exit"
    plt.waitforbuttonpress()

    
if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    main(sys.argv[1])