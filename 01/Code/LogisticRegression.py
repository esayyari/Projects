import numpy as np
import sklearn

class LogisticRegression(Transformer):
    def __init__(self,**kwrds):
        self.learning = "SGD"
        Transformer.__init__(self,**kwrds)
        
    def get_params(self,deep=False):
        return dict({"learning":self.learning,#either SGD or LBFGS
        })

    def fit(self,X,y=None,**kwrds):
        self.set_params(**kwrds)
        if self.learning=="SGD":
            self.trainSGD(X,y)
        elif self.learning=="LBFGS":
            self.trainLBFGS(X,y)
        else:
            print "Error! invalid learning method!"
            retunr None
        return self


    ######   to be completed, training code goes here ######
    
    #the SGD training function
    def trainSGD(X,y):
        n,d = X.shape
        #augment X with 1
        augX = np.concatenate([X,np.ones([n,1])],axis=1)
        #initialize parameters
        beta = 0.01 * np.random.randn([d+1,1])
        #learning rates, one for each parameter
        landa = 0.9 * np.ones([d+1,1])
        #shuffle data
        idx = np.arange(n)
        np.random.shuffle(idx)
        #main loop
        converged = False
        cnt = 0
        while not converged:
            #pick the next shuffled sample
            m = idx[cnt]
            xm = X[m,:].reshape([1,-1])
            #calculate Pm
            Pm = 1. / (1. + np.exp( -np.dot(xm,beta) ) )
            beta = beta + landa * (y[m]-Pm)*xm
            #update landa
            
            cnt = cnt+1 % n
    # the LBFGS training function
    def trainLBFGS(X,y):
        
    #convert a set of inputs to the corresponding label values
    def transform(self,X):
        
        
    
