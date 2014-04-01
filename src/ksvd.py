'''
Implements the k-SVD learning.

@author: ttsuchi
'''

from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt
import time

class KSVD:
    def __init__(self):
        self.do_plot = True
        self.P = 128
        self.K = 64
        
        self.lambdaA   = 1
        self.lambdaS   = 1
        self.sigma     = .1
        self.sigma_sal = .1
        
        self.N = 10000
        self.Np = self.K*10
        
        # Learning rate
        self.eta = .01/self.K
        
        # Number of iterations
        self.TL=15
        self.TS=10
        
        # Sampling policies
        self.policies=[self.unif, self.sala, self.salc, self.mxga, self.mxgc]
        self.NP=len(self.policies)
        self.policy_names=map(lambda f: f.__name__, self.policies)
        
        # Generate random dictionary elements
        self.A = -log(random.random((self.P, self.K)))/self.lambdaA * ((random.random((self.P, self.K)) > .5)*2-1)
        self.A /= sqrt(dot(self.A.T, self.A).diagonal())

    def learn_dictionaries(self):
        # Initial guesses
        Ag = -log(random.random((self.P, self.K)))/self.lambdaA * ((random.random((self.P, self.K)) > .5)*2-1)
        Ag = tile(Ag, (5, 1, 1))
    
        # Losses
        Lp = zeros((self.TL, self.NP))
        La = zeros((self.TL, self.NP))
    
        for tl in xrange(self.TL):
            # Generate new set of examples
            S=-log(random.random((self.K, self.N)))/self.lambdaS
            X=dot(self.A,S)+random.standard_normal((self.P, self.N))*self.sigma
            
            for np in xrange(self.NP):
                Ah=Ag[np,:,:].squeeze()
                Sh=dot(dot(diag(1/(dot(Ah.T, Ah).diagonal())), Ah.T), X)
                Sh[Sh<0]=0
                idx=self.policies[np](X, Ah, Sh)
                Xp=X[:,idx[:self.Np]]
                Ah2=self.learn(Xp,Ah)
                Lp[tl,np]=self.loss(Xp, Ah2)
                La[tl,np]=self.loss(X, Ah2)
                Ag[np,:,:]=Ah2
            
            if self.do_plot:
                plt.clf()
                plt.subplot(2,1,1)
                plt.plot(Lp[1:tl,:])    
                plt.legend(self.policy_names)
                plt.title("loss for the sampled set")
                plt.subplot(2,1,2)
                plt.plot(La[1:tl,:])
                plt.legend(self.policy_names)
                plt.title("loss for all training set")
                plt.draw()
                time.sleep(1)
            
    
    def learn(self, Xp, Ah):
        for ts in xrange(self.TS):
            Sh=dot( dot(diag(1/dot(Ah.T, Ah).diagonal()),Ah.T), Xp)
            Sh[Sh<0]=0
            Erec=dot((dot(Ah, Sh) - Xp), Sh.T)
            Espa=self.sigma**2/self.lambdaA*sign(Ah)
            Ah=Ah-self.eta*(Erec + Espa)
    
        return Ah
    
    def loss(self, X, A):
        S=dot(pinv(A),X)
        S[S<0]=0
        L=norm(X-dot(A,S),'fro')**2/(2*self.sigma**2)+sum(abs(S.flatten()))/self.lambdaS+sum(abs(A.flatten()))/self.lambdaA
        return L/X.size/self.K
    

    # Sampling policies
    # Uniform
    def unif(self, X, A, S):
        return random.permutation(X.shape[1])
    
    # Saliency across all
    def sala(self, X, A, S):
        ss=sum(abs(S),axis=0)
        ss=ss+random.standard_normal(ss.shape)/std(ss.flatten())*self.sigma_sal
        return argsort(ss)[::-1]

    # Saliency per dictionary
    def salc(self, X, A, S):
        ss=abs(S)
        ss=ss+random.standard_normal(ss.shape)/std(ss.flatten())*self.sigma_sal
        # top of 1st dictionary, top of 2nd dictionary, ... 2nd of 1st dictionary, etc.
        return argsort(ss,axis=1)[:,::-1].T.flatten()
    
    # Max gradient across all
    def mxga(self, X, A, S):
        ss=sum(abs(dot(A,S) - X),axis=0) * sum(abs(S),axis=0)
        ss=ss+random.standard_normal(ss.shape)/std(ss.flatten())*self.sigma_sal
        return argsort(ss)[::-1]

    # Max gradient per dictionary
    def mxgc(self, X, A, S):
        ss=sum(abs(dot(A,S) - X),axis=0) * abs(S)
        ss=ss+random.standard_normal(ss.shape)/std(ss.flatten())*self.sigma_sal
        # top of 1st dictionary, top of 2nd dictionary, ... 2nd of 1st dictionary, etc.
        return argsort(ss,axis=1)[:,::-1].T.flatten()

if __name__ == '__main__':
    eg=KSVD()
    eg.learn_dictionaries()
