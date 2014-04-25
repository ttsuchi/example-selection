'''
Writes figures as TikZ files.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy.random import randn
from numpy.testing import assert_allclose, assert_array_equal
import matplotlib.pyplot as plt

from algorithms.updating import GD, SPAMS

def _history(stats, column, designs):
    return matrix([stat[column].as_matrix() for stat, design in zip(stats, designs) if design.updater.__class__ == GD]).T

def plot_dist_A(multiple_stats, designs):
    design_names = [design.name() for design in designs if design.updater.__class__ == GD]
    column = 'dist_A'
    
    all_history = array([_history(stats, column, designs) for stats in multiple_stats])

    plt.figure(len(design_names) + 1, figsize = (8,6), dpi=80, facecolor='w', edgecolor='k')
    plt.clf()

    x = arange(all_history.shape[1])
    y = mean(all_history, axis = 0)
    # yerr = std(all_history, axis = 0)
    # plt.errorbar(x, y, yerr = yerr)
    plt.plot(y)
    plt.gca().set_xticklabels([])

    # Shink current axis by 20%
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    # plt.gca().legend(design_names, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().legend(design_names, loc='upper right')
    
    plt.xlabel("Iterations")
    plt.ylabel("Distance from True Dictionaries")

if __name__ == '__main__':
    import doctest
    doctest.testmod()