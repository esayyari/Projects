import numpy as np
import sklearn.linear_model as linear_model

from LogisticRegression import LogisticRegression
from readdata import read_data

X_train,y_train = read_data("./Data/train")
#X_train,y_train = X_train.toarray(),y_train.toarray()
print type(X_train)
print type(X_train)
#convert labels to 0,1
idx = np.where(y_train==-1)
y_train[idx] = 0.
lr = LogisticRegression(learning="SGD")
lr.fit(X_train,y_train)
X_test,y_test = read_data("./Data/test")
idx = np.where(y_test==-1)
y_test[idx] = 0.
print "test score:",lr.score(X_test,y_test)

sklr = linear_model.LogisticRegression(penalty="l2",C=10000.).fit(X_train,y_train)
print "sklearn.lr score:",sklr.score(X_test,y_test)
