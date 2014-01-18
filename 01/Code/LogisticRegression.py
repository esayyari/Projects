import numpy as np
import sklearn
import scipy.optimize

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
    def trainSGD(self,X,y):
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
        lclCheck = min(n,1000)#check for LCL changes after this number of iterations
        lastLCL = 0.
        stopLCLChng = .001
        while not converged:
            #pick the next shuffled sample
            m = idx[cnt]
            xm = X[m,:].reshape([1,-1])
            #calculate Pm
            Pm = 1. / (1. + np.exp( -np.dot(xm,beta) ) )
            beta = beta + landa * (y[m]-Pm)*xm
            #update landa
            cnt = cnt+1 % n
            if cnt==lclcheck:
                Pm = 1./ (1. + np.exp(-np.dot(X,beta)))
                LCL = y * np.log(Pm) + (1.-y) * np.log(1.-Pm)
                LCL = LCL.sum()
                if np.abs(LCL-lastLCL) < stopLCLChng:
                    converged = True

    # the LBFGS training function
    def trainLBFGS(self,X,y):
        n,d = X.shape
        #augment X with 1
        augX = np.concatenate([X,np.ones([n,1])],axis=1)
        #initialize parameters
        beta = 0.01 * np.random.randn([d+1,1])
        #call lbfgs_b
        scipy.optimize.fmin_l_bfgs_b(self.LCL, beta, fprime=self.LCLderiv, args=(augX,y), iprint=0)
        
    # beta is a column vector of size d
    def LCL(self,beta,*args):
        X = args[0]# n x d
        y = args[1]# n x 1
        Pm = 1./ (1. + np.exp(-np.dot(X,beta)))
        lcl = y * np.log(Pm) + (1.-y) * np.log(1.-Pm)
        return lcl.sum()
        
    #convert a set of inputs to the corresponding label values
    def transform(self,X):
        
        
    
