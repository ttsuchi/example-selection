#! /usr/bin/env python
'''
Creates plots.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from importlib import import_module
from inc.design import Experiment

import re
import os, sys
from os.path import splitext, basename
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib2tikz import save

FIGURES_DIR = '../figures/'

def main(figname, name, subname, tikz = False, pdf = True):
    figures_module = import_module("analyses.figures")
    plot_fn = getattr(figures_module, "plot_" + figname)

    pattern = name + '-' + subname + '[0-9]+\.pkl'
    multiple_stats = []
    designs = None
    for experiment in [Experiment.load(splitext(basename(f))[0]) for f in os.listdir(Experiment.SAVE_DIR) if re.match(pattern, f)]:
        print "Loaded %s, ended at iteration = %d" % (experiment.name, experiment.itr)
        multiple_stats.append(experiment.stats)
        designs = experiment.designs
    
    if designs is None:
        raise Exception("No designs matched " + pattern)

    plot_fn(multiple_stats, designs)
    plt.draw()
    
    if pdf:
        pdf_filename = FIGURES_DIR + name + '-' + figname + '.pdf'
        pp = PdfPages(pdf_filename)
        pp.savefig(plt.gcf())
        pp.close()
        print "Saved to %s" % pdf_filename

    if tikz:
        tikz_filename = FIGURES_DIR + name + '-' + figname + '.tikz'
        save(tikz_filename,
                  figureheight = '\\figureheight',
                  figurewidth = '\\figurewidth')
        print "Saved to %s" % tikz_filename

    sys.stdout.write("\n\nDone, close the figures to exit\n");
    sys.stdout.flush()
    plt.waitforbuttonpress()
    
if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))

    import argparse
    
    parser = argparse.ArgumentParser(description='Plots the result of dictionary learning experiments defined under the experiment package.')
    parser.add_argument('figname',         default='dist_A', help='name of the figure to create')
    parser.add_argument('name',            help='experiment name corresponding to the module name under the "experiment" package')
    parser.add_argument('subname',         default='', nargs='?',  help='experiment sub-name, used to distinguish the save files')
    parser.add_argument('-t','--tikz',                      default=False, help='saves Tikz file', action='store_true')
    parser.add_argument('-p','--pdf',                      default=True, help='saves PDF file', action='store_true')

    args = parser.parse_args()
    main(**vars(args))