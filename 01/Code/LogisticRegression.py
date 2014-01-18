import numpy as np
import sklearn
import scipy.optimize
import scipy.sparse
from transformer import Transformer
from sklearn.preprocessing import StandardScaler

class LogisticRegression(Transformer):
    def __init__(self,**kwrds):
        self.learning = "SGD"
        Transformer.__init__(self,**kwrds)
    
    #parameters set and get
    def set_params(self,**kwrds):
        for k in kwrds.keys():
            if k=="learning":
                self.learning = kwrds[k]
    
    def get_params(self,deep=False):
        return dict({"learning":self.learning,#either SGD or LBFGS
        })

    #fit function
    def fit(self,X,y=None,**kwrds):
        self.set_params(**kwrds)
        if self.learning=="SGD":
            self.trainSGD(X,y)
        elif self.learning=="LBFGS":
            self.trainLBFGS(X,y)
        else:
            print "Error! invalid learning method!"
            return None
        return self


    ######   to be completed, training code goes here ######
    
    #the SGD training function
    def trainSGD(self,X,y):
        pass

    # the LBFGS training function
    def trainLBFGS(self,X,y):
        n,d = X.shape
        ones = np.ones([n,1])
        #augment X with 1
        augX = np.concatenate([X.toarray(),ones],axis=1)
        self.scaler = StandardScaler().fit(augX)
        augX = self.scaler.transform(augX)
        self.X_train = augX
        self.y_train = y
        #initialize parameters
        beta = 0.0001 * np.random.randn(d+1)
        ret = scipy.optimize.fmin_l_bfgs_b(self.LCL, beta,fprime=self.LCLderiv,pgtol=1e-5)
        print type(ret)
        #store the parameters
        self.beta = ret[0]
        #print "params:",ret
        
    # beta is a column vector of size d
    def LCL(self,beta,*args):
        X = self.X_train# n x d
        y = self.y_train# n x 1
        Pm = 1. / (1. + np.exp(-np.dot(X,beta)))
        #print Pm.min(),Pm.max()
        lcl = y * np.log(Pm) + (1.-y) * np.log(1.-Pm)
        return -lcl.sum()
        
    def LCLderiv(self,beta,*args):
        X = self.X_train# n x d
        y = self.y_train# n x 1
        n,d = X.shape
        Pm = 1. / (1. + np.exp(-np.dot(X,beta)))
        t1 = y-Pm
        temp = np.tile(t1.reshape([-1,1]),[1,d]) *X
        temp = -np.sum(temp,axis=0)
        #print temp.shape
        return temp
        
    #convert a set of inputs to the corresponding label values
    def transform(self,X):
        n,d = X.shape
        augX = np.concatenate([X.toarray(),np.ones([n,1])],axis=1)
        augX = self.scaler.transform(augX)
        Pm = 1./ (1. + np.exp(-np.dot(augX,self.beta)))
        decision = Pm>=0.5
        rslt = np.zeros(decision.shape)
        rslt[np.where(decision==True)]=1.
        return rslt

    def score(self,X,y):
        y_p = self.transform(X)
        print y_p
        return np.mean(y_p==y)
