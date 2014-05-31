#! /usr/bin/env python
'''
Creates plots.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from importlib import import_module
from inc.design import Experiment
import analyses.figures

import re
import os, sys
from os.path import splitext, basename
from sets import Set
import string, operator, difflib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib2tikz import save

FIGURES_DIR = '../figures/'

def _mstr(a,b):
    m = difflib.SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
    return a[m.a:m.a+m.size]

def main(figname, name, subname, collect = [], styles = [], tikz = False, pdf = True):
    figures_module = import_module("analyses.figures")
    plot_fn = getattr(figures_module, "plot_" + figname)
    collect.sort()
    styles.sort()
    collect_fns = [getattr(figures_module, 'collect_' + collect_name) for collect_name in collect]
    style_fns = [getattr(figures_module, 'style_' + style_name) for style_name in styles]

    names = [name]
    if len(subname) > 0:
        names.append(subname)
    pattern = string.join(names, '-') + '[0-9]*\.pkl'
    multiple_stats = []
    designs = None
    max_iter = -1
    for experiment in [Experiment.load(splitext(basename(f))[0]) for f in os.listdir(Experiment.SAVE_DIR) if re.match(pattern, f)]:
        print "Loaded %s, ended at iteration = %d" % (experiment.name, experiment.itr)
        if experiment.itr >= max_iter:
            max_iter = experiment.itr
            multiple_stats.append((experiment.stats, experiment.As, experiment.Xs))
            designs = experiment.designs
        else:
            print "Skipping data, since it's less than %d" % max_iter
    
    if designs is None:
        raise Exception("No designs matched " + pattern)
    
    # Convert the data structure to be indexed by design
    data = [{'design': design,
             'stats': [stats[i] for stats, _, _ in multiple_stats],
             'As':    [As[i]    for _, As, _    in multiple_stats],
             'Xs':    [Xs[i]    for _, _, Xs    in multiple_stats]
             } for i, design in enumerate(designs)]

    # Only collect data for certain designs
    data = [d for d in data if reduce(operator.and_, [f(d['design']) for f in collect_fns], True)]

    # Clean design names
    design_names = [d['design'].name() for d in data]
    common_part = reduce(lambda l, s: _mstr(l,s), design_names, design_names[-1])
    for d in data:
        d['name'] = string.join([s for s in d['design'].name().split(common_part) if len(s)>0],'-')

    plot_fn(data, style_fns)
    plt.draw()
    
    file_name = string.join([FIGURES_DIR + name, figname] + collect + styles, '-')
    
    if pdf:
        pdf_filename = file_name + '.pdf'
        pp = PdfPages(pdf_filename)
        pp.savefig(plt.gcf())
        pp.close()
        print "Saved to %s" % pdf_filename

    if tikz:
        tikz_filename = file_name + '.tikz'
        save(tikz_filename,
                  figureheight = '\\figureheight',
                  figurewidth = '\\figurewidth',
                  extra = Set([
                        'y tick label style={/pgf/number format/.cd, precision=3, fixed, 1000 sep={}}','scaled y ticks=false',
                        'x tick label style={/pgf/number format/.cd, precision=3, fixed, 1000 sep={}}','scaled x ticks=false',
                        ]))
        print "Saved to %s" % tikz_filename

    sys.stdout.write("\n\nDone, close the figures to exit\n");
    sys.stdout.flush()
    plt.waitforbuttonpress()
    
if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))

    plots = [f[len('plot_'):] for f in dir(analyses.figures) if f.startswith('plot_')]
    collects = [f[len('collect_'):] for f in dir(analyses.figures) if f.startswith('collect_')]
    styles = [f[len('style_'):] for f in dir(analyses.figures) if f.startswith('style_')]

    import argparse
    
    parser = argparse.ArgumentParser(description='Plots the result of dictionary learning experiments defined under the experiment package.')
    parser.add_argument('figname',         default=plots[0], help='name of the figure to create', choices=plots)
    parser.add_argument('name',            help='experiment name corresponding to the module name under the "experiment" package')
    parser.add_argument('subname',         default='', nargs='?',  help='experiment sub-name, used to distinguish the save files')
    parser.add_argument('-t','--tikz',                      default=False, help='saves Tikz file', action='store_true')
    parser.add_argument('-p','--pdf',                      default=False, help='saves PDF file', action='store_true')
    parser.add_argument('-c','--collect',  action='append', metavar='C', help='Collects only the stats identitified by C. {' + string.join(collects, ',') + '}', default=[], choices=collects)
    parser.add_argument('-s','--styles',   action='append', metavar='S', help='Use the line style identifiedy by S. {' + string.join(styles, ',') + '}', default=[], choices=styles)

    args = parser.parse_args()
    main(**vars(args))