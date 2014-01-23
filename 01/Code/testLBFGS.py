import numpy as np
import sklearn.linear_model as linear_model

from LogisticRegression import LogisticRegression
from readdata import read_data
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

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
parameters = dict({
    "clf__weight_decay"      : [.001,.01,.1,1,10,100,1000],
#    "clf__init_beta"         : [0.,0.00001,.0001,0.001,.01,.1,1.,10],
                  })

lr = LogisticRegression(learning="LBFGS",X_test=X_test,y_test=y_test)
pipeline = Pipeline([
                        ("clf"  ,lr),
                    ])
f = GridSearchCV(pipeline,parameters,n_jobs=-1,verbose=-1,cv=3)
f.fit(X_train,y_train)
best_parameters = f.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
print "score on test:",f.score(X_test,y_test)
print("Best score: %0.3f" % f.best_score_)


sklr = linear_model.LogisticRegression(penalty="l2",C=10000.)
parameters = dict({
    "clf__C"         : [.001,.01,.1,1,10,100,1000],
    "clf__penalty"      : ["l1","l2"],
                  })

ppln = Pipeline([
                        ("clf"  ,sklr),
                    ])
f = GridSearchCV(ppln,parameters,n_jobs=-1,verbose=-1,cv=10)
f.fit(X_train,y_train)
print "score on test:",f.score(X_test,y_test)
print("Best score: %0.3f" % f.best_score_)
#print "sklearn.lr score:",sklr.score(X_test,y_test)
