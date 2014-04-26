'''
Writes figures as TikZ files.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy.random import randn
from numpy.testing import assert_allclose, assert_array_equal
import matplotlib.pyplot as plt
import string

from algorithms.updating import GD, SPAMS

def _chosen(design):
    return design.updater.__class__ == GD

def _names(designs):
    return [string.split(design.name(),'-')[0] for design in designs if _chosen(design)]

def _sorted_design_names(design_names, by):
    return [l for l,_ in sorted(zip(design_names, by), key=lambda x: x[1])]

def _history(stats, column, designs):
    return matrix([stat[column].as_matrix() for stat, design in zip(stats, designs) if _chosen(design)]).T

def plot_dist_A(multiple_stats, designs):
    design_names = _names(designs)
    column = 'dist_A'
    
    all_history = array([_history(stats, column, designs) for stats in multiple_stats])

    plt.figure(len(design_names) + 1, figsize = (8,6), dpi=80, facecolor='w', edgecolor='k')
    plt.clf()

    x = arange(all_history.shape[1])
    y = mean(all_history, axis = 0)
    yerr = std(all_history, axis = 0)
    
    # Sort by the last performance
    sort_data = -mean(all_history[:,-1,:].squeeze(), axis=0)
    idx = argsort(sort_data)
    y = y[:, idx]
    yerr = yerr[:, idx]
    design_names = _sorted_design_names(design_names, sort_data)
    
    for yind in range(y.shape[1]):
        #plt.fill_between(xcb, lcb, ucb, alpha=0.3, facecolor='gray')
        ym = y[:, yind]
        base_line, = plt.plot(ym)
        plt.fill_between(x, ym - yerr[:, yind], ym + yerr[:, yind], facecolor=base_line.get_color(), alpha = 0.05)
    plt.gca().set_xscale('log')

    # Shink current axis by 20%
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    # plt.gca().legend(design_names, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().legend(design_names, loc='lower left')
    
    plt.xlabel("Iterations")
    plt.ylabel("Distance from True Dictionaries")

def plot_dist_A_bar(multiple_stats, designs):
    design_names = _names(designs)
    column = 'dist_A'
    
    all_history = array([_history(stats, column, designs) for stats in multiple_stats])

    plt.figure(len(design_names) + 1, figsize = (8,6), dpi=80, facecolor='w', edgecolor='k')
    plt.clf()

    data = all_history[:,-1,:].squeeze()
    x = arange(data.shape[1])
    y = mean(data, axis=0)
    yerr = std(data, axis = 0)
    ind = arange(x.shape[0])
    
    idx = argsort(-y)
    
    width = .8
    plt.bar(ind, y[idx], width, color = 'b', yerr=yerr[idx])
    plt.gca().set_xticks(ind+width/2)
    plt.gca().set_xticklabels(_sorted_design_names(design_names, -y))
    
    plt.ylabel("Distance from True Dictionaries")

def plot_snr(multiple_stats, designs):
    design_names = _names(designs)
    column = 'mean_Xsnr_p'

    all_history = array([_history(stats, column, designs) for stats in multiple_stats]).squeeze()
    start = int(ceil(.6 * all_history.shape[0]))
    data = all_history[start:, :].squeeze()
    
    ind = arange(data.shape[1])
    width = .8
    plt.bar(ind, mean(data, axis=0), width, color = 'b', yerr = std(data, axis=0))
    plt.gca().set_xticks(ind+width/2)
    plt.gca().set_xticklabels(design_names)
    plt.ylabel("SNR of selected examples [dB]")

if __name__ == '__main__':
    import doctest
    doctest.testmod()