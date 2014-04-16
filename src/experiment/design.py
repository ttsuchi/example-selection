'''
Executes a dictionary learning experiment.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy.testing import assert_allclose, assert_array_equal

import matplotlib.pyplot as plt
import itertools
import string
import pandas
import pickle

from data.dictionary import Random, to_image
from algorithms.learning import update_with

from common import mtr

import datetime
import time

from joblib import Parallel, delayed
from tempfile import mkdtemp
import os
import os.path as path


class Experiment(object):
    def __init__(self, name, observables, selectors = None, encoders = None, updaters = None, designs = None, **kwds):
        self.name = name
        self.observables = observables
        self.Astar =  self.observables.dictionary.A if hasattr(self.observables, 'dictionary') else None

        if designs is None:
            designs = [Design(self, selector, encoder, updater) for selector, encoder, updater in itertools.product(selectors, encoders, updaters)]
        self.designs = designs

    def run(self, num_iter, save_every = 10, plot_every = 10, parallel_jobs = 1):
        """Executes the dictionary learning experiment.
        """
        
        # Don't plot on a headless environment
        do_plot = os.environ.has_key('DISPLAY') and plot_every > 0
        
        state = State(self)
        
        for _ in range(num_iter):
            state.update(parallel_jobs)

            if save_every > 0 and (state.itr % save_every == 1):
                state.save()

            if do_plot and (state.itr % plot_every == 1):
                state.plot()

            print("iter=%3d / %3d, %f[s] elapsed, estimated finish at %s\n" % 
                  (state.itr, num_iter, state.elapsed, state.estimated_finish(num_iter)))

        state.save()

        if do_plot:
            print "Done, close the figures to exit"
            plt.waitforbuttonpress()


class Design(object):
    """Describes a particular experiment variation with specific algorithms.
    """
    
    def __init__(self, experiment, selector, encoder, updater, **kwds):
        self.experiment = experiment
        self.selector = selector
        self.encoder = encoder
        self.updater = updater       

    def name(self):
        objs = []
        if len(set(map(lambda design: design.selector, self.experiment.designs))) > 1:
            objs.append(self.selector)
        if len(set(map(lambda design: design.encoder, self.experiment.designs))) > 1:
            objs.append(self.encoder)
        if len(set(map(lambda design: design.updater, self.experiment.designs))) > 1:
            objs.append(self.updater)
        return string.join([obj.__class__.__name__ for obj in objs], '-')

class State:
    """Represents a state in the dictionary learning iteration.
    """
    
    SAVE_DIR='../results/'
    
    def __init__(self, experiment):
        self.experiment = experiment

        # Initial dictionary set
        A = Random(experiment.observables.p, experiment.observables.K, sort = False).A
        
        self.As     = [mtr(A.copy()) for _ in experiment.designs]
        self.Xs     = []
        self.losses = [pandas.DataFrame() for _ in experiment.designs]
        self.stats  = pandas.DataFrame() 
        self.itr    = 0
        self.elapsed = 0.0

    def update(self, parallel_jobs = 1):
        """Performs the dictionary update step.
        """
        start = time.time()
        
        # Generate mini-batches
        X = self.experiment.observables.sample()

        # Execute in parallel or serial        
        designs = self.experiment.designs
        if parallel_jobs > 1:
            X_filename = path.join(mkdtemp(), 'X.dat')
            Xm = memmap(X_filename, dtype = X.dtype, mode='w+', shape = X.shape)
            Xm[:] = X[:]
            results = Parallel(n_jobs = parallel_jobs)(delayed(update_with)(design, Xm, A) for A, design in zip(self.As, designs))
        else:
            results = [update_with(design, X, A) for A, design in zip(self.As, designs)]        
        
        self.As, current_losses, self.Xs = tuple(map(list, zip(*results)))
        self.losses = map( lambda l_c: l_c[0].append(l_c[1], ignore_index = True), zip(self.losses, current_losses))
        
        self.elapsed = (time.time() - start)
        self.stats = self.stats.append({'elapsed': self.elapsed}, ignore_index = True)
        self.itr += 1

    def save(self):
        with open(State.SAVE_DIR + self.experiment.name + '.pkl', 'wb') as fout:
            pickle.dump(self, fout)

    @classmethod
    def load(cls, name):
        with open(State.SAVE_DIR + name + '.pkl', 'r') as fin:
            return pickle.load(fin)

    def estimated_finish(self, num_iter):
        mean_elapsed = self.stats['elapsed'].mean()
        estimated_finish = (datetime.datetime.now() + datetime.timedelta(0, mean_elapsed*(num_iter - self.itr - 1)))
        return estimated_finish.strftime('%x %X')

    def history(self, column):
        return matrix(map(lambda loss: loss[column].as_matrix(), self.losses)).T

    def plot(self):
        design_names = map(lambda design: design.name(), self.experiment.designs)
        Astar = self.experiment.Astar
       
        N = 2
        N+= 0 if Astar is None else 1
        for p, (selector_name, A, X) in enumerate(zip(design_names, self.As, self.Xs)):
            plt.figure(p + 1, figsize = (8,6), dpi=80, facecolor='w', edgecolor='k')
            plt.clf()
            n=1
            
            if Astar is not None:
                plt.subplot(1,N,1)
                plt.imshow(to_image(Astar), aspect = 'equal', interpolation = 'nearest', vmin = 0, vmax = 1)
                plt.axis('off')
                plt.title('Ground-truth')
                n+=1
                
            plt.subplot(1,N,n)
            plt.imshow(to_image(A), aspect = 'equal', interpolation = 'nearest', vmin = 0, vmax = 1)
            plt.axis('off')
            plt.title('Learned dictionary')
            n+=1
            
            plt.subplot(1,N,n)
            plt.imshow(to_image(X), aspect = 'equal', interpolation = 'nearest', vmin = 0, vmax = 1)
            plt.axis('off')
            plt.title('Top selected examples')
            
            plt.suptitle(selector_name)
            plt.draw()

        N = 4
        N+= 0 if Astar is None else 1
        plt.figure(len(design_names) + 1, figsize = (8,6), dpi=80, facecolor='w', edgecolor='k')
        plt.clf()
        
        plt.subplot(N,1,1)
        plt.plot(self.history('loss_sampled'))
        plt.gca().set_xticklabels([])
        plt.title("loss for the sampled set")
    
        plt.subplot(N,1,2)
        plt.plot(self.history('loss_all'))
        plt.gca().set_xticklabels([])
        plt.title("loss for all training set")
    
        plt.subplot(N,1,3)
        plt.plot(self.history('std'))
        if Astar is not None:
            plt.gca().set_xticklabels([])
        plt.title("mean difference in A")

        if Astar is not None:
            plt.subplot(N,1,4)
            plt.plot(self.history('conformity'))
            plt.title("Average conformity")
        
        plt.subplot(N,1,N)
        plt.plot(zeros((2,len(design_names))))
        plt.axis('off')
        plt.legend(design_names, loc='center', ncol=4)

        plt.draw()
        
        plt.pause(1)
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()
