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
import os

from data.dictionary import Random, to_image
from algorithms.learning import update_with
from inc.common import mtr
from inc.execution import Serial

import datetime
import time

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

class Experiment(object):
    """Represents a particular state in the dictionary learning iteration.
    """
    
    SAVE_DIR='../results/'
    
    def __init__(self, name, generator, selectors = None, encoders = None, updaters = None, designs = None, **kwds):
        self.name = name
        self.generator = generator
        self.Astar =  self.generator.dictionary.A if hasattr(self.generator, 'dictionary') else None

        if designs is None:
            designs = [Design(self, selector, encoder, updater) for selector, encoder, updater in itertools.product(selectors, encoders, updaters)]
        self.designs = designs

        # Initial dictionary set with some example sets
        A = generator.generate()[:,:generator.K]
        
        self.As     = [mtr(A.copy()) for _ in designs]
        self.Xs     = []
        self.losses = [pandas.DataFrame() for _ in designs]
        self.stats  = pandas.DataFrame() 
        self.itr    = 0
        self.elapsed = 0.0

    def run(self, num_iter, executor):
        """Executes the dictionary learning experiment.
        """
        
        while self.itr < num_iter:
            start = time.time()
            
            # Generate mini-batches
            X = self.generator.generate()
            assert all(isfinite(X))

            # Perform the update
            results = executor(update_with, X, self.As, self.designs, self.itr)
            
            self.As, current_losses, self.Xs = tuple(map(list, zip(*results)))
            self.losses = map( lambda l_c: l_c[0].append(l_c[1], ignore_index = True), zip(self.losses, current_losses))
            
            self.elapsed = (time.time() - start)
            self.stats = self.stats.append({'elapsed': self.elapsed}, ignore_index = True)
            self.itr += 1

            yield self

    def save(self):
        file_name = Experiment.SAVE_DIR + self.name + '.pkl'
        with open(file_name, 'wb') as fout:
            pickle.dump(self, fout)
        return file_name

    @classmethod
    def load(cls, name):
        file_name = Experiment.SAVE_DIR + name + '.pkl'
        if os.path.isfile(file_name):
            with open(file_name, 'r') as fin:
                return pickle.load(fin)
        else:
            return None

    def estimated_finish(self, num_iter):
        mean_elapsed = self.stats['elapsed'].mean()
        estimated_finish = (datetime.datetime.now() + datetime.timedelta(0, mean_elapsed*(num_iter - self.itr - 1)))
        return estimated_finish.strftime('%x %X')

    def history(self, column):
        return matrix(map(lambda loss: loss[column].as_matrix(), self.losses)).T

    def plot(self):
        design_names = map(lambda design: design.name(), self.designs)
        Astar = self.Astar
       
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
            plt.plot(self.history('distance'))
            plt.title("average distance from the true dictionary")
        
        plt.subplot(N,1,N)
        plt.plot(zeros((2,len(design_names))))
        plt.axis('off')
        plt.legend(design_names, loc='center', ncol=4)

        plt.draw()
        
        plt.pause(1)
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()
