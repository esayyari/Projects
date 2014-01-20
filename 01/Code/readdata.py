import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split

def read_data(data_file_name, normalize=True, permute=False):
    X, y = load_svmlight_file(data_file_name)
    if normalize:
        X=scale(X,0,False)
    if permute:
         X, y = shuffle(X, y, random_state=0)
    idx = np.where(y==-1)
    y[idx] = 0.
    return (X,y)

def LR_SGD_grid(X, y, fold=4, range_eta=[1e-2, 1e-1, 1e0], range_eta_dacey= [0.01, 0.1, 0.5, 0.9]):
    best_acc, best_eta, best_edecay=0, None, None
    with open('LR.SGD.grid', 'w') as f:
        for eta in range_eta:
            for edecay in range_eta_dacey:
                acc=cross_validation(X, y, fold, eta, edecay, "SGD")
                if acc>best_acc:
                    best_acc, best_eta, best_edecay=acc, eta, edecay
                f.write('{0}\t{1}\t{2}\n'.format(eta,edecay,acc))
                print 'CV Accuracy: {0}  eta: {1}  eta_decay: {2}'.format(acc, eta, edecay)
        f.write('****************\n{0}\t{1}\t{2}\n'.format(best_acc, best_eta, best_edecay))
    print 'Best CV Accuracy: {0}  Best eta: {1}  Best eta_decay: {2}'.format(best_acc, best_eta, best_edecay)
    return best_eta, best_edecay
            

def RLR_SGD_grid(X, y, fold, ranges=None):
    pass

def cross_validation(X,y,fold, eta, edecay, solver):
    from sklearn.cross_validation import StratifiedKFold
    from LogisticRegression import LogisticRegression
    scores=[]
    skf = StratifiedKFold( y, fold) 
    for train_index, test_index in skf:
        X_train, X_test, y_train, y_test = X[train_index,:], X[test_index,:], y[train_index], y[test_index]
        lr = LogisticRegression(learning=solver,eta_decay=edecay,eta_0=eta)
        lr.fit(X_train,y_train)
        scores.append(lr.score(X_test,y_test))
    return np.mean(scores)