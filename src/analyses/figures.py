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

from algorithms.encoding import LASSO, SOMP, OMP, KSparse
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

def collect_only_omp(design):
    return design.encoder.__class__ == OMP

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

TRANSLATION_TABLE = {
    'Unif':'Uniform', 'ErrS': 'ErrBySum', 'ErrD': 'ErrByElement', 'MXGS': 'GradBySum', 'MXGD': 'GradByElement',
    'SNRS': 'SNRBySum', 'SNRD': 'SNRByElement', 'SUNS':'SUNBySum', 'SUND':'SUNByElement'
}

def _translate_name(name):
    return TRANSLATION_TABLE[name] if TRANSLATION_TABLE.has_key(name) else name

# Style schemes

def style_by_sd(line, group, design):
    if group == (0,0):
        plt.setp(line, color='k', linewidth=3.0)
    elif group[0] == 1:
        plt.setp(line, ls='dotted')
    elif group[0] == 2:
        plt.setp(line, ls='solid')
        
def style_for_talk(line, group, design):
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

PAPER_COLORS = {
    'Unif': '#000000',
    'ErrS': '#EE1111',
    'ErrD': '#EE1111',
    'MXGS': '#EE11EE',
    'MXGD': '#EE11EE',
    'SUNS': '#1111EE',
    'SUND': '#1111EE',
    'SNRS': '#11EE88',
    'SNRD': '#11EE88',
    'SalMap': '#333333'
}

def style_for_paper(line, group, design):
    if group == (0,0):
        plt.setp(line, color='k', linewidth=3.0)
        return

    plt.setp(line, linewidth=2.0)

    if group[0] == 0 or group[0] == 1: # Sum
        plt.setp(line, ls='dashed')
    elif group[0] == 2: # PerD
        plt.setp(line, ls='solid')

    selector_name = design.selector.__class__.__name__
    if PAPER_COLORS.has_key(selector_name):
        plt.setp(line, color = PAPER_COLORS[selector_name])

def _plot_by_epoch(data, style_fns, stat_name, ylabel):
    N = len(data)
    for d in data:
        stat = array([stat[stat_name].as_matrix().T for stat in d['stats']])  # run-by-time
        d['mean'] = nanmean(stat, axis=0)
        d['sem']  = nanstd(stat,  axis=0) / sqrt(N)
        d['last'] = nanmean(stat, axis=0)[-1]

    # Sort by the last performance
    data = sorted(data, key=lambda d: d['last'], reverse=True)

    plt.figure(figsize = (9.7,6), dpi=72, facecolor='w', edgecolor='k')
    # Plot
    for d in data:
        ym = d['mean']
        yerr = d['sem'] * 1.96
        x = arange(len(ym))+1
        idx=isfinite(ym)
        ym  = ym[idx]
        yerr= yerr[idx]
        x   = x[idx]
        base_line, = plt.plot(x,ym)
        
        for style_fn in style_fns:
            style_fn(base_line, d['design'].selector.group(), d['design'])
        
        #plt.fill_between(x, ym - yerr, ym + yerr, facecolor=base_line.get_color(), alpha = 0.05)
    plt.gca().set_xscale('log')

    # Shink current axis by 20%
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    # plt.gca().legend(design_names, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().legend([_translate_name(d['name']) for d in data], loc='lower left', bbox_to_anchor=(1, 0))
    
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    _set_fonts()

def plot_dist_A(data, style_fns):
    _plot_by_epoch(data, style_fns, 'dist_A', 'Distance from True Dictionaries')

def plot_dist_S(data, style_fns):
    _plot_by_epoch(data, style_fns, 'dist_S', 'Distance from True Activations')

def plot_vqd(data, style_fns):
    _plot_by_epoch(data, style_fns, 'vqd', 'D')


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

def plot_dist_A_all(data, style_fns):
    N = len(data)
    for d in data:
        stat = array([stat['dist_A'].as_matrix().T for stat in d['stats']])  # run-by-time
        d['mean'] = nanmean(stat, axis=0)
        d['sem']  = nanstd(stat,  axis=0) / sqrt(N)
        d['last'] = nanmean(stat, axis=0)[-1]


    kdata = filter(lambda d: d['design'].encoder.__class__.__name__ == 'KSparse', data)
    kdata = sorted(kdata, key=lambda d: d['last'], reverse=True)
    sorder = map(lambda d: d['design'].selector.__class__.__name__, kdata)
    
    y=zeros((3, len(sorder)))
    encoders = ['KSparse', 'SOMP', 'LASSO']
    for i, e in enumerate(encoders):
        for j, s in enumerate(sorder):
            y[i, j] = next(d['last'] for d in data if d['design'].encoder.__class__.__name__ == e and d['design'].selector.__class__.__name__ == s)

    plt.figure(figsize = (9.7,6), dpi=72, facecolor='w', edgecolor='k')
    # Plot
    for i, d in enumerate(kdata):        
        base_line, = plt.plot(arange(3), y[:,i], marker = 'o')
        
        for style_fn in style_fns:
            style_fn(base_line, d['design'].selector.group(), d['design'])        
        
    plt.gca().set_xlim([-.5, 2.5])
    plt.gca().set_xticks(arange(3))
    plt.gca().set_xticklabels(['k-Sparse', 'L0', 'L1'])
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

def _plot_bars(data, style_fns, stat_name, label, horizontal = True):
    for d in data:
        dist_A = array([stat['dist_A'].as_matrix().T for stat in d['stats']])  # run-by-time
        d['last'] = nanmean(dist_A, axis=0)[-1]
        s = array([stat[stat_name].as_matrix().T for stat in d['stats']])  # run-by-time
        d['mean'] = nanmean(nanmean(s, axis=0)[-10:])    
        d['std'] = nanmean(nanstd(s, axis=0)[-10:])
    
    # Sort by the last performance
    data = sorted(data, key=lambda d: d['last'])

    y = array([d['mean'] for d in data])
    yerr = array([d['std'] for d in data])

    ind = arange(len(y))
    width = .8

    if horizontal:
        bars = plt.barh(ind, y, width, color = 'b')
    else:
        bars = plt.bar(ind, y, width, color = 'b', yerr = yerr)

    _set_fonts()

    for bar, d in zip(bars, data):
        for style_fn in style_fns:
            style_fn(bar, d['design'].selector.group(), d['design'])

    if horizontal:
        plt.gca().set_yticks(ind+width/2)
        plt.gca().set_yticklabels([_translate_name(d['name']) for d in data])
        plt.xlabel(label)
    else:
        plt.gca().set_xticks(ind+width/2)
        plt.gca().set_xticklabels([_translate_name(d['name']) for d in data])
        plt.ylabel(label)


def plot_snr(data, style_fns):
    _plot_bars(data, style_fns, 'mean_Xsnr_p', "SNR of selected examples [dB]", True)

def plot_xp_dist(data, style_fns):
    _plot_bars(data, style_fns, 'mean_Xp_dist', "Mean distance among selected examples", True)

def plot_vqd_bar(data, style_fns):
    _plot_bars(data, style_fns, 'vqd', 'D(X_N||X_n)', True)

if __name__ == '__main__':
    import doctest
    doctest.testmod()