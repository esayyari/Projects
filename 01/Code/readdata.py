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

def LR_SGD_grid(X, y, fold=4, range_eta=[1e-3, 1e-2, 1e-1, 1e0, 1e1]):
    best_acc, best_eta=0, None
    with open('LR.SGD.grid', 'w') as f:
        for eta in range_eta:
            acc=cross_validation(X, y, fold, eta, "SGD")
            if acc>best_acc:
                best_acc, best_eta=acc, eta
            f.write('{0}\t{1}\n'.format(eta,acc))
            print 'CV Accuracy: {0}  eta: {1}'.format(acc, eta)
        f.write('****************\n{0}\t{1}\n'.format(best_eta, best_acc))
    print 'Best eta: {0}  Best CV Accuracy: {1}'.format(best_eta, best_acc)
    return best_eta
            

def RLR_SGD_grid(X, y, fold=4, range_weight_decay=[1e-2, 1e-1, 1e0, 1e1, 1e2], range_eta= [1e-3, 1e-2, 1e-1, 1e0, 1e1]):
# def RLR_SGD_grid(X, y, fold=4, range_weight_decay=[1e2], range_eta= [1e-1, 1e0, 1e1]):
    best_acc, best_eta, best_weight=0, None, None
    with open('RLR.SGD.grid', 'w') as f:
        for eta in range_eta:
            for weight in range_weight_decay:
                acc=cross_validation(X, y, fold, eta, "SGD", weight)
                if acc>best_acc:
                    best_acc, best_eta, best_weight=acc, eta, weight
                f.write('{0}\t{1}\t{2}\n'.format(eta,weight,acc))
                print 'eta: {0}  Weight decay: {1}  CV Accuracy: {2}'.format(eta, weight, acc)
        f.write('****************\n{0}\t{1}\t{2}\n'.format(best_eta, best_weight, best_acc))
    print 'Best eta: {0}  Best Weight decay: {1}  Best CV Accuracy: {2}'.format(best_eta, best_weight, best_acc)
    return best_eta, best_weight

def cross_validation(X,y,fold, eta, solver="SGD", wdecay=0):
    from sklearn.cross_validation import StratifiedKFold
    from LogisticRegression import LogisticRegression
    scores=[]
    skf = StratifiedKFold( y, fold) 
    for train_index, test_index in skf:
        X_train, X_test, y_train, y_test = X[train_index,:], X[test_index,:], y[train_index], y[test_index]
        lr = LogisticRegression(learning=solver,weight_decay=wdecay,eta_0=eta)
        lr.fit(X_train,y_train)
        scores.append(lr.score(X_test,y_test))
    return np.mean(scores)