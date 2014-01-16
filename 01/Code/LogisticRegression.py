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
        
    # the LBFGS training function
    def trainLBFGS(X,y):
        
    #convert a set of inputs to the corresponding label values
    def transform(self,X):
        
        
    
