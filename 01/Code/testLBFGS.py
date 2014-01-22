import numpy as np
import sklearn.linear_model as linear_model

from LogisticRegression import LogisticRegression
from readdata import read_data
from sklearn import cross_validation


X_train,y_train = read_data("./Data/train")
X_train = X_train.toarray()
#convert labels to 0,1
idx = np.where(y_train==-1)
y_train[idx] = 0.
kf = cross_validation.KFold(y_train.shape[0], n_folds=10, indices=False)
X_test,y_test = read_data("./Data/test")
X_test = X_test.toarray()
idx = np.where(y_test==-1)
y_test[idx] = 0.
for train, test in kf:
    X_tn, X_ts, y_tn, y_ts = X_train[train], X_train[test], y_train[train], y_train[test]
    lr = LogisticRegression(learning="LBFGS")
    lr.fit(X_tn,y_tn)
    print "kfold score:",lr.score(X_ts,y_ts)
    print "test score:",lr.score(X_test,y_test)

sklr = linear_model.LogisticRegression(penalty="l2",C=10000.).fit(X_train,y_train)
print "sklearn.lr score:",sklr.score(X_test,y_test)
