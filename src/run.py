#! /usr/bin/env python
'''
Runs the experiment.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from importlib import import_module
from inc.design import Experiment
from inc.execution import Serial, Parallel

import matplotlib.pyplot as plt

def main(name, subname, num_iter, parallel_jobs, save_every, plot_every):
    experiment_name = name + '-' + subname
    
    # Either load the existing experiment or create a new one
    experiment = Experiment.load(experiment_name)
    if experiment is None:
        print "Creating %s" % experiment_name
        experiment_module = import_module("experiment." + name)
        create = getattr(experiment_module, 'create')
        experiment = create(experiment_name)
    else:
        print "Loaded %s, starting with iteration = %d" % (experiment_name, experiment.itr)

    if parallel_jobs == 1:
        executor = Serial()
    else:
        executor = Parallel(parallel_jobs)
    
    # Don't plot on a headless environment
    do_plot = os.environ.has_key('DISPLAY') and plot_every > 0
   
    # Run the experiment up to num_iter
    for state in experiment.run(num_iter, executor):
        if save_every > 0 and (state.itr % save_every == 1):
            save_file = state.save()
            print "saving to %s" % save_file

        if do_plot and (state.itr % plot_every == 1):
            state.plot()

        print("iter=%3d / %3d, %f[s] elapsed, estimated finish at %s" % 
              (state.itr, num_iter, state.elapsed, state.estimated_finish(num_iter)))

    experiment.save()

    if do_plot:
        experiment.plot()
        print "Done, close the figures to exit"
        plt.waitforbuttonpress()
    
if __name__ == '__main__':
    import os, sys
    os.chdir(os.path.dirname(sys.argv[0]))

    import argparse
    
    parser = argparse.ArgumentParser(description='Executes the dictionary learning experiments defined under the experiment package.')
    parser.add_argument('name',            help='experiment name corresponding to the module name under the "experiment" package')
    parser.add_argument('subname',         nargs='?',           default='',   help='experiment sub-name, used to distinguish the save files')
    parser.add_argument('num_iter',        nargs='?', type=int, default=1000, help='number of iterations')
    parser.add_argument('--parallel_jobs', nargs='?', type=int, default=1,  metavar='P', help='executes using P processes')
    parser.add_argument('--save_every',    nargs='?', type=int, default=10, metavar='N', help='save every N iterations')
    parser.add_argument('--plot_every',    nargs='?', type=int, default=10, metavar='N', help='plot every N iterations')

    args = parser.parse_args()
    main(**vars(args))