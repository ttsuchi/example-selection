'''
Writes figures as TikZ files.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy.random import randn
from numpy.testing import assert_allclose, assert_array_equal
import matplotlib.pyplot as plt
import string

from data.dictionary import to_image

from algorithms.encoding import LASSO, SOMP, KSparse
from algorithms.updating import GD, SPAMS
from algorithms.selection import *

# Filters
def collect_all(design):
    return True

def collect_only_gd(design):
    return design.updater.__class__ == GD

def collect_only_spams(design):
    return design.updater.__class__ == SPAMS

def collect_only_ksparse(design):
    return design.encoder.__class__ == KSparse

def collect_only_lasso(design):
    return design.encoder.__class__ == LASSO

def collect_only_somp(design):
    return design.encoder.__class__ == SOMP

def collect_for_talk(design):
    return design.selector.__class__ in [Unif, UsedD, MXGS, MXGD, SalMap, SUNS, SUND, ErrS, SNRD, SNRS]

# Plots

def _chosen(design):
    return design.updater.__class__ == GD

def _cidx(designs):
    return nonzero(array([_chosen(design) for design in designs]))[0]

def _names(designs):
    design_names = [design.name() for design in designs if _chosen(design)]
    return [(string.split(name,'-')[0] if '-' in name else name) for name in design_names]

def _sorted_design_names(design_names, by):
    return [l for l,_ in sorted(zip(design_names, by), key=lambda x: x[1])]

def _history(stats, column, designs):
    return matrix([stat[column].as_matrix() for stat, design in zip(stats, designs) if _chosen(design)]).T

def _set_fonts():
    fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'], 'weight' : 'light'}
    a = plt.gca()
    a.set_xticklabels(a.get_xticks(), fontProperties)
    a.set_yticklabels(a.get_yticks(), fontProperties)

# Style schemes

def style_by_sd(line, group):
    if group == (0,0):
        plt.setp(line, color='k', linewidth=3.0)
    elif group[0] == 1:
        plt.setp(line, ls='dotted')
    elif group[0] == 2:
        plt.setp(line, ls='solid')
        
def style_for_talk(line, group):
    if group == (0,0):
        plt.setp(line, color='k', linewidth=3.0)
        return

    plt.setp(line, linewidth=2.0)

    if group[0] == 0 or group[0] == 1: # Sum
        plt.setp(line, ls='dashed')
    elif group[0] == 2: # PerD
        plt.setp(line, ls='solid')

    if group[1] == 0: # Heuristic
        plt.setp(line, color='#C82506')
    elif group[1] == 1: # Saliency
        plt.setp(line, color='#0365C0')#'#0B5D18')


def plot_dist_A(data, style_fns):
    # Calculate stats
    N = len(data)
    for d in data:
        dist_A = array([stat['dist_A'].as_matrix().T for stat in d['stats']])  # run-by-time
        d['mean'] = mean(dist_A, axis=0)
        d['sem']  = std(dist_A,  axis=0) / sqrt(N)
        d['last'] = mean(dist_A, axis=0)[-1]

    # Sort by the last performance
    data = sorted(data, key=lambda d: d['last'], reverse=True)

    plt.figure(figsize = (9.7,6), dpi=72, facecolor='w', edgecolor='k')
    # Plot
    for d in data:
        #plt.fill_between(xcb, lcb, ucb, alpha=0.3, facecolor='gray')
        ym = d['mean']
        yerr = d['sem'] * 1.96
        x = arange(len(ym))+1
        base_line, = plt.plot(x,ym)
        
        for style_fn in style_fns:
            style_fn(base_line, d['design'].selector.group())
        
        plt.fill_between(x, ym - yerr, ym + yerr, facecolor=base_line.get_color(), alpha = 0.05)
    plt.gca().set_xscale('log')

    # Shink current axis by 20%
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    # plt.gca().legend(design_names, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().legend([d['name'] for d in data], loc='lower left', bbox_to_anchor=(1, 0))
    
    plt.xlabel("Iterations")
    plt.ylabel("Distance from True Dictionaries")
    _set_fonts()

def plot_dist_A_bar(data, style_fns):
    for d in data:
        dist_A = array([stat['dist_A'].as_matrix().T for stat in d['stats']])  # run-by-time
        d['std']  = std(dist_A,  axis=0)[-1]
        d['last'] = mean(dist_A, axis=0)[-1]

    # Sort by the last performance
    data = sorted(data, key=lambda d: d['last'], reverse=True)
    
    y = array([d['last'] for d in data])
    yerr = array([d['std'] for d in data])
    
    ind=arange(len(data))
    width = .8
    plt.bar(ind, y, width, color = 'b', yerr=yerr)
    plt.gca().set_xticks(ind+width/2)
    plt.gca().set_xticklabels([d['name'] for d in data])
    
    plt.ylabel("Distance from True Dictionaries")
    _set_fonts()

def plot_true_dictionaries(data, style_fns):
    Astar = data[0]['design'].experiment.Astar
    plt.imshow(to_image(Astar), aspect = 'equal', interpolation = 'nearest', vmin = 0, vmax = 1)
    plt.axis('off')

def plot_sample_X0(data, style_fns):
    X = data[0]['Xs'][0]
    plt.imshow(to_image(X), aspect = 'equal', interpolation = 'nearest', vmin = 0, vmax = 1)
    plt.axis('off')

def plot_best_dictionaries(data, style_fns):
    for d in data:
        dist_A = array([stat['dist_A'].as_matrix().T for stat in d['stats']])  # run-by-time
        d['last'] = mean(dist_A, axis=0)[-1]

    # Sort by the last performance
    data = sorted(data, key=lambda d: d['last'], reverse=True)
        
    plt.figure(figsize = (6,6), dpi=72, facecolor='w', edgecolor='k')
    print "Best design: %s" % data[-1]['name']

    A = data[-1]['As'][0]
    plt.imshow(to_image(A), aspect = 'auto', interpolation = 'nearest', vmin = 0, vmax = 1)
    plt.axis('off')
    
def plot_worst_dictionaries(data, style_fns):
    for d in data:
        dist_A = array([stat['dist_A'].as_matrix().T for stat in d['stats']])  # run-by-time
        d['last'] = mean(dist_A, axis=0)[-1]

    # Sort by the last performance
    data = sorted(data, key=lambda d: d['last'], reverse=True)
        
    plt.figure(figsize = (6,6), dpi=72, facecolor='w', edgecolor='k')
    print "Worst design: %s" % data[0]['name']

    A = data[0]['As'][0]
    plt.imshow(to_image(A), aspect = 'auto', interpolation = 'nearest', vmin = 0, vmax = 1)
    plt.axis('off')

def plot_snr(data, style_fns):
    for d in data:
        dist_A = array([stat['dist_A'].as_matrix().T for stat in d['stats']])  # run-by-time
        d['last'] = mean(dist_A, axis=0)[-1]
        snr = array([stat['mean_Xsnr_p'].as_matrix().T for stat in d['stats']])  # run-by-time
        d['mean_snr'] = mean(snr, axis=0)[-1]    
        d['std_snr'] = mean(snr, axis=0)[-1]    
    
    # Sort by the last performance
    data = sorted(data, key=lambda d: d['last'])

    y = array([d['mean_snr'] for d in data])
    yerr = array([d['std_snr'] for d in data])

    ind = arange(len(y))
    width = .8
    horizontal = True

    if horizontal:
        bars = plt.barh(ind, y, width, color = 'b')
    else:
        bars = plt.bar(ind, y, width, color = 'b', yerr = yerr)

    _set_fonts()

    for bar, d in zip(bars, data):
        for style_fn in style_fns:
            style_fn(bar, d['design'].selector.group())

    if horizontal:
        plt.gca().set_yticks(ind+width/2)
        plt.gca().set_yticklabels([d['name'] for d in data])
        plt.xlabel("SNR of selected examples [dB]")
    else:
        plt.gca().set_xticks(ind+width/2)
        plt.gca().set_xticklabels([d['name'] for d in data])
        plt.ylabel("SNR of selected examples [dB]")



if __name__ == '__main__':
    import doctest
    doctest.testmod()