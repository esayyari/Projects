import numpy as np
import sklearn.linear_model as linear_model
from LogisticRegression import LogisticRegression
from readdata import *

doGridSearch= not  False
regularized=True
shuffle_examples, normalize_fetures= True , True
X_train,y_train = read_data("./Data/train",normalize_fetures,shuffle_examples)
X_test,y_test   = read_data("./Data/test",normalize_fetures)

if doGridSearch:
    if regularized:
        eta, weight= RLR_SGD_grid(X_train, y_train)
    else:
        eta= LR_SGD_grid(X_train, y_train)
else:
    eta=0.01
    if regularized:
        weight=1
    else:
        weight=0

solver="SGD"
# lr = LogisticRegression(learning=solver)
lr = LogisticRegression(learning=solver, eta_0=eta, weight_decay=weight, max_epoch=2, batch_size=1)
lr.fit(X_train,y_train)

sklr = linear_model.LogisticRegression(penalty="l2",C=10000.).fit(X_train,y_train)

print solver, "Test Accuracy:",lr.score(X_test,y_test)
print "SkitLearn Accuracy:",sklr.score(X_test,y_test)

