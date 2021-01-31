import numpy as np
import pandas as pd
import sys

def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)
    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
     
    return X, X_mean, X_std

def _train_dev_split(X, Y, dev_ratio = 0.25):
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X 
    return np.round(_f(X, w, b)).astype(np.int)

def _f(X, w, b):
    return _sigmoid(np.matmul(X, w) + b)

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def _sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

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

def gen(X_train, Y_train, X_dev, Y_dev):
    train_size = X_train.shape[0]
    dev_size = X_dev.shape[0]
    test_size = X_test.shape[0]
    data_dim = X_train.shape[1]

    # Compute in-class mean
    X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
    X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])
    mean_0 = np.mean(X_train_0, axis = 0)
    mean_1 = np.mean(X_train_1, axis = 0)  

    # Compute in-class covariance
    cov_0 = np.zeros((data_dim, data_dim))
    cov_1 = np.zeros((data_dim, data_dim))

    for x in X_train_0:
        cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
    for x in X_train_1:
        cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

    # Shared covariance is taken as a weighted average of individual in-class covariance.
    cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])
    u, s, v = np.linalg.svd(cov, full_matrices=False)
    inv = np.matmul(v.T * 1 / s, u.T)
    w = np.dot(inv, mean_0 - mean_1)
    b =  (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1))\
    + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0])

    return w,b 

def out(ans, name):
    with open(name, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(ans):
            f.write('%d,%d\n' %(i+1, v))

if __name__ == "__main__":

    X_train = np.genfromtxt(sys.argv[3], delimiter=',',encoding = "utf8")
    Y_train = np.genfromtxt(sys.argv[4], delimiter=',')
    X_test = np.genfromtxt(sys.argv[5], delimiter=',', skip_header=1)
    X_train = X_train[1:]

    title = np.genfromtxt("model/title", delimiter=',',encoding = "utf8",dtype = "str")
    title_drop = list(title[64:105])
    X_train = pd.DataFrame(X_train, columns = title)
    X_test  = pd.DataFrame(X_test, columns = title)
    X_train = preprocess(X_train, title)
    X_test = preprocess(X_test, title)

    X_train, X_mean, X_std = _normalize(X_train, train = True)
    X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)
    dev_ratio = 0.1
    X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)

    w, b = gen(X_train, Y_train, X_dev, Y_dev)
    predictions = 1 - _predict(X_test, w, b)
    out(predictions, sys.argv[6])