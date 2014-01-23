import numpy as np
import sklearn
import scipy.optimize
import scipy.sparse
from transformer import Transformer
from sklearn.preprocessing import StandardScaler

class LogisticRegression(Transformer):
    def __init__(self,**kwrds):
        self.learning       = "SGD"
        self.eta            =0.01
        self.eta_decay      =0.9
        self.beta_decay     =0
        self.max_epoc       =10
        self.batch_size     =1
        self.eps            =1e-6
        self.weight_decay   =1
        self.init_beta      = 0.1
        Transformer.__init__(self,**kwrds)
    
    #parameters set and get
    def set_params(self,**kwrds):
        for k in kwrds.keys():
            if k=="learning":
                self.learning = kwrds[k]
            elif k=="eta_0":
                self.eta = kwrds[k]
            elif k=="eta_decay":
                self.eta_decay = kwrds[k]
            elif k=="beta_decay":
                self.beta_decay = kwrds[k]
            elif k=="max_epoch":
                self.max_epoch = kwrds[k]
            elif k=="batch_size":
                self.batch_size = kwrds[k]
            elif k=="weight_decay":
                self.weight_decay = kwrds[k]
            elif k=="lcl":
                self.lcl_lbfgs = kwrds[k]
            elif k=="lcltest":
                self.lcltest_lbfgs = kwrds[k]
            elif k=="X_test":
                self.X_test = kwrds[k]
            elif k=="y_test":
                self.y_test = kwrds[k]
            elif k=="init_beta":
                self.init_beta = kwrds[k]

    def get_params(self,deep=False):
        return dict({"learning":self.learning,#either SGD or LBFGS
                     "lcl":self.lcl_lbfgs,
                     "lcltest":self.lcltest_lbfgs,
                     "X_test":self.X_test,
                     "y_test":self.y_test,
                     "weight_decay":self.weight_decay,
                     "init_beta":self.init_beta,
        })

    #fit function
    def fit(self,X,y=None,**kwrds):
        self.pos=np.nonzero(y == 1)
        self.neg=np.nonzero(y != 1)
        self.lcl = np.ones(X.shape[0])
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
        n,d = X.shape
        ones = np.ones([n,1])
        #augment X with 1
        augX = np.concatenate([X.toarray(),ones],axis=1)
        self.scaler = StandardScaler().fit(augX)
        augX = self.scaler.transform(augX)
        self.X_train = augX
        self.y_train = y
        #initialize parameters
        self.beta = 0.0001 * np.random.randn(d+1)
        obj=self.LCL(self.beta)
#         print obj
#         print 'epoch: {0} LCL: {1}'.format(0, obj)
        if self.weight_decay:
            fname=str(self.batch_size)+'RLR.log'
        else:
            fname=str(self.batch_size)+'LR.log'
        with open(fname, 'w') as f:
            f.write('epoch\tobj\tnormg\tnormb ')
            for epoch in range(self.max_epoc):
                for iter in range(n):
                    g=self.LCLderiv(self.beta,self.batch_size, iter)
                    f.write('{0}\t{1}\t{2}\t{3}\n'.format( epoch, obj, np.linalg.norm(g),np.linalg.norm(self.beta) ))
                    self.beta-= self.eta*g
                    obj=self.LCL(self.beta)
                    
                    if obj==float('inf'):
    #                     print "Epoch: {0}  Iter: {1}  Objective: {2}".format(epoch, iter , obj) , np.linalg.norm(self.beta), np.linalg.norm(g)
                        return
    #                 print 'iter: {0} LCL: {1}'.format(iter, obj)
                
                self.eta*=self.eta_decay
                obj=self.LCL(self.beta)
    #             print 'norm_beta',np.linalg.norm(self.beta) ,'obj',obj
    #             if np.linalg.norm(beta, ord=inf) <0.001:
    #                 beta=np.zeros(d+1)
    #                 break
                
    #             print 'epoch: {0} LCL: {1}'.format(epoch+1, obj)
            

     # the LBFGS training function
    def trainLBFGS(self,X,y):
        n,d = X.shape
        ones = np.ones([n,1])
        #augment X with 1
        augX = np.concatenate([X,ones],axis=1)
        self.scaler = StandardScaler().fit(augX)
        augX = self.scaler.transform(augX)
        self.X_train = augX
        self.y_train = y
        #initialize parameters
        beta = self.init_beta * np.random.randn(d+1)
        ret = scipy.optimize.fmin_l_bfgs_b(self.LCL, beta,fprime=self.LCLderiv,#factr=1e17,
                                              callback=self.bookkeeping)
        #print type(ret)
        #store the parameters
        self.beta = ret[0]
        #print self.beta
        #print "params:",ret
        
    # beta is a column vector of size d
    def LCL(self,beta,*args):
        X = self.X_train# n x d
        y = self.y_train# n x 1
        n,d=X.shape
        Pm = 1. / (1. + np.exp(-np.dot(X,beta)))
#         print "***********", Pm.min(),Pm.max()
#         self.lcl = y * np.log(Pm) + (1.-y) * np.log(1.-Pm) #Mohsen's (not efficient)

        self.lcl[self.pos] = np.log(Pm[self.pos])
        self.lcl[self.neg] =  np.log(1.-Pm[self.neg])
        return -self.lcl.sum() +self.weight_decay* np.linalg.norm(beta)
        
    def LCLderiv(self,beta,batch_size=None, index=None):
        if batch_size == None:
            X = self.X_train# n x d
            y = self.y_train# n x 1
        else:
            X = self.X_train[index:index+batch_size,:]# n x d
            y = self.y_train[index:index+batch_size]# n x 1
        n,d = X.shape
        Pm = 1. / (1. + np.exp(-np.dot(X,beta)))
        t1 = y-Pm
        g = np.tile(t1.reshape([-1,1]),[1,d]) *X
        g = -np.sum(g,axis=0)+2.*self.weight_decay*beta
        return g
        
    #convert a set of inputs to the corresponding label values
    def transform(self,X):
        n,d = X.shape
        if type(X)==np.ndarray:
            augX = np.concatenate([X,np.ones([n,1])],axis=1)
        else:
            augX = np.concatenate([X.toarray(),np.ones([n,1])],axis=1)
        augX = self.scaler.transform(augX)
        Pm = 1./ (1. + np.exp(-np.dot(augX,self.beta)))
        decision = Pm>=0.5
        rslt = np.zeros(decision.shape)
        rslt[np.where(decision==True)]=1.
        return rslt

    def score(self,X,y):
        y_p = self.transform(X)
        return np.mean(y_p==y)
        
    def bookkeeping(self,beta,*args):
        self.lcl_lbfgs.append(self.LCL(beta))
        self.lcltest_lbfgs.append(self.lclTest(beta))
#        self.betanorm.append(np.sqrt(np.square(beta).sum()))
       # beta is a column vector of size d

    def lclTest(self,beta,*args):
        X = self.X_test# n x d
        y = self.y_test# n x 1
        n,d = X.shape
        ones = np.ones([n,1])
        #augment X with 1
        augX = np.concatenate([X,ones],axis=1)
        augX = self.scaler.transform(augX)
        X = augX
        Pm = 1. / (1. + np.exp(-np.dot(X,beta)))
        #print Pm.min(),Pm.max()
        lcl = y * np.log(Pm) + (1.-y) * np.log(1.-Pm)
        return -lcl.sum() + self.weight_decay * np.sqrt(np.square(beta).sum())
 
        
