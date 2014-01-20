import numpy as np
import sklearn.linear_model as linear_model
from LogisticRegression import LogisticRegression
from readdata import read_data, LR_SGD_grid

doGridSearch=False
shuffle_examples, normalize_fetures= True , True
X_train,y_train = read_data("./Data/train",normalize_fetures,shuffle_examples)
X_test,y_test   = read_data("./Data/test",normalize_fetures)

if doGridSearch:
    eta,edecay= LR_SGD_grid(X_train, y_train)
else:
    eta,edecay=0.01, 0.9
solver="SGD"
# lr = LogisticRegression(learning=solver)
lr = LogisticRegression(learning=solver, eta_0=eta, eta_decay=edecay, max_epoch=2, batch_size=10)
lr.fit(X_train,y_train)

sklr = linear_model.LogisticRegression(penalty="l2",C=10000.).fit(X_train,y_train)

print solver, "Test Accuracy:",lr.score(X_test,y_test)
print "SkitLearn Accuracy:",sklr.score(X_test,y_test)

