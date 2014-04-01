'''
Executes a dictionary learning experiment.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from numpy.random import randn
from numpy.testing import assert_allclose, assert_array_equal

from scipy.io import savemat

from data.observable import FromDictionaryL1
from data.dictionary import Random

class Experiment(object):
    def __init__(self, name, observables, selectors, encoder, updater, num_iter = 200, **kwds):
        self.name = name
        self.observables = observables
        self.selectors = selectors
        self.encoder = encoder
        self.updater = updater
        self.num_iter = num_iter

    @property
    def params(self):
        return {
            'observables': self.observables.params,
            'selectors'  : [selector.params for selector in self.selectors],
            'encoder'    : self.encoder.params,
            'updater'    : self.updater.params
            }

    def evaluate_loss(self, X, A, Xp):
        # TODO
        loss = 0
        A = 0
        return loss, A

    def run(self, As = None, plot = False, plot_every = 10, save_every = 10):
        Np = len(self.selectors)
        
        if As == None:
            A = Random(self.observable.dictionary.p, self.observable.dictionary.K).A()
            As = [A.copy()] * Np
        
        losses = zeros((self.num_iter, Np))
        
        for itr in range(self.num_iter):
            # Generate mini-batches
            X = self.observables.sample()
            for i, selector in enumerate(self.selectors):
                A = As[i]
                S = self.encoder.encode(X, A)
                
                # Pick examples to learn from
                idx = selector.select(X, A, S)
                Xp = X[:, idx]
                
                # Update dictionary using these examples
                A = self.updater.update(Xp, A)
                
                # Evaluate the loss (and reorder A, if necessary)
                loss, A = self.evaluate_loss(X, A, Xp)
                
                # Save results
                As[i] = A
                losses[itr, Np] = loss
            
            if plot and mod(itr, plot_every) == 1:
                # TODO do plot
                pass
            
            if mod(itr, save_every) == 1:
                results = {
                    'As': As,
                    'losses': losses
                    }
                results.update(self.params())
                savemat(self.name + '.mat', results)

            print("iter=%3d / %3d, %f[s] elapsed, estimated finish at %s\n" % (itr,self.num_iter))#,ttt,datestr(now + ttt*(ITR-itr)/(60*60*24))))

if __name__ == '__main__':
    Experiment().run()
