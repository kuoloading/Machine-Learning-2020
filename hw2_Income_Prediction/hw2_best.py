
import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import GradientBoostingClassifier
import pickle

def _normalize_column_normal(X, train=True, specified_column = None, X_mean=None, X_std=None):
    if train:
        if specified_column == None:
            specified_column = np.arange(X.shape[1])
        length = len(specified_column)
        X_mean = np.reshape(np.mean(X[:, specified_column],0), (1, length))
        X_std  = np.reshape(np.std(X[:, specified_column], 0), (1, length))
    
    X[:,specified_column] = np.divide(np.subtract(X[:,specified_column],X_mean), X_std)
     
    return X, X_mean, X_std

def preprocess(data, title):
    title_drop = list(title[64:105])
    data = pd.DataFrame(data, columns = title)
    data = data.drop(columns=title_drop)
    from numpy import inf
    data["capital_gain^2"] = data["capital_gain"] * data["capital_gain"] 
    data["fnlwgt^2"] = data["fnlwgt"] * data["fnlwgt"] 
    data["capital_loss^2"] = data["capital_loss"] * data["capital_loss"] 
    data["hours_per_week^2"] = data["hours_per_week"] * data["hours_per_week"] 
    data["age^2"] = data["age"] * data["age"] 
    data["age^3"] = data["age"] * data["age"] * data["age"] 
    data["capital_gain^3"] = data["capital_gain"] * data["capital_gain"] * data["capital_gain"] 
    data["capital_gain^4"] = data["capital_gain"] * data["capital_gain"] * data["capital_gain"] * data["capital_gain"]
    data["age^4"] = data["age"] * data["age"] * data["age"] * data["age"] 

    data["capital_gain_log"] = np.log(data["capital_gain"].replace(0, np.nan))
    data["age_log"] = np.log(data["age"].replace(0, np.nan))
    data["capital_loss_log"] = np.log(data["capital_loss"].replace(0, np.nan))
    data["hours_per_week_log"] = np.log(data["hours_per_week"].replace(0, np.nan))
    data["fnlwgt_log"] = np.log(data["fnlwgt"].replace(0, np.nan))

    data[data == -inf] = 0
    data = data.fillna(0)
    data  = data.to_numpy()
    return data

def out(ans, name):
    with open(name, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(ans):
            f.write('%d,%d\n' %(i+1, v))

def GBC(X_train, Y_train):
    clf = GradientBoostingClassifier(n_estimators=550, learning_rate=0.2, random_state=42, min_samples_split=1550, min_samples_leaf=15, max_depth=4, max_features='sqrt').fit(X_train, Y_train.ravel())
    clf.fit(X_train, Y_train)
    return clf

if __name__ == "__main__": 
    X_train_fpath =sys.argv[3]
    Y_train_fpath = sys.argv[4]
    X_test_fpath = sys.argv[5]
    X_train = np.genfromtxt(X_train_fpath, delimiter=',',encoding = "utf8")
    Y_train = np.genfromtxt(Y_train_fpath, delimiter=',')
    X_test = np.genfromtxt(X_test_fpath, delimiter=',', skip_header=1)
    X_train = X_train[1:]

    title = np.genfromtxt("model/title", delimiter=',',encoding = "utf8",dtype = "str")
    title_drop = list(title[64:105])
    X_train = pd.DataFrame(X_train, columns = title)
    X_test  = pd.DataFrame(X_test, columns = title)
    X_train = preprocess(X_train, title)
    X_test = preprocess(X_test, title)

    col = [0, 1,3, 4, 5, 65, 66, 67, 68, 69, 70, 71, 72, 73,74 ,75 ,76,77, 78] #刪掉國籍用的
    X_train, X_mean, X_std = _normalize_column_normal(X_train, specified_column=col)
    X_test, _, _= _normalize_column_normal(X_test, train=False, specified_column = col, X_mean=X_mean, X_std=X_std)
    # gb_clf = GBC(x_train_add, np.reshape(y_train,(-1,)))
    # with open('model/clf.pickle', 'wb') as f:
    #     pickle.dump(clf, f)

    with open('model/clf.pickle', 'rb') as f:
        clf = pickle.load(f)
    ans = clf.predict(X_test)
    out(ans, sys.argv[6])