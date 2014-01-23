import numpy as np
import sklearn.linear_model as linear_model
from LogisticRegression import LogisticRegression
from readdata import *

doGridSearch=   True
regularized=   True
bsize=1
for regularized in [True, False]:
    for bsize in [1, 10, 100]:
        shuffle_examples, normalize_fetures= True , True
        X_train,y_train = read_data("./Data/train",normalize_fetures,shuffle_examples)
        X_test,y_test   = read_data("./Data/test",normalize_fetures)
        
        if doGridSearch:
            if regularized:
                eta, weight, cv_acc= RLR_SGD_grid(X_train, y_train,bsize)
            else:
                eta, cv_acc= LR_SGD_grid(X_train, y_train, bsize)
                weight=0
        else:
            eta=0.01
            if regularized:
                weight=0.01
            else:
                weight=0
        solver="SGD"
        # lr = LogisticRegression(learning=solver)
        lr = LogisticRegression(learning=solver, eta_0=eta, weight_decay=weight, max_epoch=10, batch_size=bsize)
        lr.fit(X_train,y_train)
        
        sklr = linear_model.LogisticRegression(penalty="l2",C=10000.).fit(X_train,y_train)
        test_acc=lr.score(X_test,y_test)
        runname=str(bsize)+('','R')[regularized]+'LR'
        print solver, runname, "Test Accuracy:", test_acc
        with open('out','a') as f:
            f.write('{0}\t\t{1}\t\t{2}\n'.format(runname, cv_acc, test_acc))

# print "SkitLearn Accuracy:",sklr.score(X_test,y_test)

