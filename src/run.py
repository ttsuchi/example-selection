#! /usr/bin/env python
'''
Runs the experiment.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from importlib import import_module
from inc.design import Experiment
from inc.execution import Serial, IParallel

import matplotlib.pyplot as plt

def main(name, subname, num_iter, save_every, plot_every, parallel = False, clean = False):
    experiment_name = name + '-' + subname
    
    # Either load the existing experiment or create a new one
    experiment = None if clean else Experiment.load(experiment_name)
    if experiment is None:
        print "Creating %s" % experiment_name
        experiment_module = import_module("experiment." + name)
        create = getattr(experiment_module, 'create')
        experiment = create(experiment_name)
    else:
        print "Loaded %s, starting with iteration = %d" % (experiment_name, experiment.itr)

    if parallel:
        executor = IParallel()
    else:
        executor = Serial()
    
    # Don't plot on a headless environment
    do_plot = os.environ.has_key('DISPLAY') and plot_every > 0
   
    # Run the experiment up to num_iter
    for state in experiment.run(num_iter, executor):
        if save_every > 0 and (state.itr % save_every == 1):
            state.save()

        if do_plot and (state.itr % plot_every == 1):
            state.plot()

        sys.stdout.write("iter=%3d / %3d, %f[s] elapsed, estimated finish at %s\r" % 
              (state.itr, num_iter, state.elapsed, state.estimated_finish(num_iter)))
        sys.stdout.flush()

    experiment.save()

    if do_plot:
        experiment.plot()
        sys.stdout.write("\n\nDone, close the figures to exit\n");
        sys.stdout.flush()
        plt.waitforbuttonpress()
    
if __name__ == '__main__':
    import os, sys
    os.chdir(os.path.dirname(sys.argv[0]))

    import argparse
    
    parser = argparse.ArgumentParser(description='Executes the dictionary learning experiments defined under the experiment package.')
    parser.add_argument('name',            help='experiment name corresponding to the module name under the "experiment" package')
    parser.add_argument('subname',         nargs='?',           default='',   help='experiment sub-name, used to distinguish the save files')
    parser.add_argument('num_iter',        nargs='?', type=int, default=1000, help='number of iterations')
    parser.add_argument('-p','--parallel',                      default=False, help='executes using IPython.parallel', action='store_true')
    parser.add_argument('-c','--clean',                         default=False, help='starts the experiment from fresh', action='store_true')
    parser.add_argument('--save_every',    nargs='?', type=int, default=10, metavar='N', help='save every N iterations')
    parser.add_argument('--plot_every',    nargs='?', type=int, default=10, metavar='N', help='plot every N iterations')

    args = parser.parse_args()
    main(**vars(args))