import pandas as pd
import numpy as np
import sys

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

def _normalize_column_normal(X, train=True, specified_column = None, X_mean=None, X_std=None):
    if train:
        if specified_column == None:
            specified_column = np.arange(X.shape[1])
        length = len(specified_column)
        X_mean = np.reshape(np.mean(X[:, specified_column],0), (1, length))
        X_std  = np.reshape(np.std(X[:, specified_column], 0), (1, length))
    
    X[:,specified_column] = np.divide(np.subtract(X[:,specified_column],X_mean), X_std)
    return X, X_mean, X_std

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])
    
def train_dev_split(X, y, dev_size=0.2):
    train_len = int(round(len(X)*(1-dev_size)))
    return X[0:train_len], y[0:train_len], X[train_len:None], y[train_len:None]

def _sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-6, 1-1e-6)

def get_prob(X, w, b):
    # the probability to output 1
    return _sigmoid(np.add(np.matmul(X, w), b))

def infer(X, w, b):
    # use round to infer the result
    return np.round(get_prob(X, w, b))

def _cross_entropy(y_pred, Y_label):
    # compute the cross entropy
    cross_entropy = -np.dot(Y_label, np.log(y_pred))-np.dot((1-Y_label), np.log(1-y_pred))
    return cross_entropy

def _gradient(X, Y_label, w, b):
    # return the mean of the graident
    y_pred = get_prob(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.mean(np.multiply(pred_error.T, X.T), 1)
    b_grad = -np.mean(pred_error)
    return w_grad, b_grad

def _gradient_regularization(X, Y_label, w, b, lamda):
    # return the mean of the graident
    y_pred = get_prob(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.mean(np.multiply(pred_error.T, X.T), 1)+lamda*w
    b_grad = -np.mean(pred_error)
    return w_grad, b_grad

def _loss(y_pred, Y_label, lamda, w):
    return _cross_entropy(y_pred, Y_label) + lamda * np.sum(np.square(w))

def train(X_train, Y_train):
    dev_size = 0.1155
    X_train, Y_train, X_dev, Y_dev = train_dev_split(X_train, Y_train, dev_size = dev_size)
    w = np.zeros((X_train.shape[1],)) 
    b = np.zeros((1,))
    regularize = True
    if regularize:
        lamda = 0.001 #0.00005
    else:
        lamda = 0
    max_iter = 100  # iteration 超過 200沒有意義！
    batch_size = 8
    learning_rate = 8
    num_train = len(Y_train)
    num_dev = len(Y_dev)
    step =1
    loss_train = []
    loss_validation = []
    train_acc = []
    dev_acc = []
    
    for epoch in range(max_iter):
        if (epoch% 15 == 0):
            learning_rate  = learning_rate/2 
        X_train, Y_train = _shuffle(X_train, Y_train)
        
        total_loss = 0.0

        for idx in range(int(np.floor(len(Y_train)/batch_size))):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]
            
            w_grad, b_grad = _gradient_regularization(X, Y, w, b, lamda)
            w = w - learning_rate/np.sqrt(step) * w_grad
            b = b - learning_rate/np.sqrt(step) * b_grad
            step += 1
    return w, b  # return loss for plotting

def out(ans, name):
    with open(name, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(ans):
            f.write('%d,%d\n' %(i+1, v))

if __name__ == '__main__':

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
    col = [0, 1,3, 4, 5, 65, 66, 67, 68, 69, 70, 71, 72, 73,74 ,75 ,76,77, 78] #刪掉國籍用的
    X_train, X_mean, X_std = _normalize_column_normal(X_train, specified_column=col)
    
    w, b = train(X_train, Y_train)
    X_test, _, _= _normalize_column_normal(X_test, train=False, specified_column = col, X_mean=X_mean, X_std=X_std)
    result = infer(X_test, w, b)
    out(result, sys.argv[6])