import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


def sigmoid(x):
    return np.array(np.exp(x) / (np.exp(x) + 1.))

class LogisticRegressionClassifier:
    def __init__(self, learning_rate = 0.00001, eta = 0.001, iters = 2000):
        self.learning_rate = learning_rate
        self.eta = eta
        self.iters = iters  
        self.wei = []
        self.col = []  
    
    def fit(self, X, y):
        Y = pd.get_dummies(y) 
        X = np.array(X)
        size = X.shape[0] # batch_size
        num = X.shape[1] # n
        self.col = Y.columns.tolist()
        l = len(self.col)
        self.wei = np.zeros((l, X.shape[1]))
        Y = np.array(Y) # (b, l)

        for i in range(self.iters):
            sum = 0.
            wx = np.zeros((1, l))         
            for j in range(size):

                x = X[j]
                y = Y[j]
                x = x.reshape(1, -1)
                y = y.reshape(1, -1)

                wx = np.dot(self.wei, x.T).T #(1, l)
                # wx /= num

                grad = np.dot(y.T, x) - np.dot(x.T, sigmoid(wx)).T #(1, n)

                self.wei += 1. * self.learning_rate * grad
                sum += np.sum(np.abs(grad))
                # print('sum is ', sum)
                # print('grad is', grad)

            if sum / size < self.eta:
                break
            if i % 50 == 0:
                print('iter ', i, ' times cost is ', sum/size)


    def predict(self, X):
        res = [self.col[
            np.argmax(sigmoid(np.dot(x, self.wei.T)))
        ] for x in X]
        
        return res
    
    def score(self, X, y):
        res = self.predict(X)
        # print(res)
        # print(y)
        sc = 0.
        for i in range(len(y)):
            if res[i] == y[i]:
                sc += 1
        return sc / len(y)


def load_data(path = 'Input/train.csv'):
    raw_data = pd.read_csv(path)
    y = raw_data['label'].values
    # raw_data.drop(['label'], axis=1)
    del raw_data['label']
    X = raw_data.values
    return X, y



if __name__ == "__main__":
    print('Start read data')
    time_1 = time.time()
    X, y = load_data()
    # 没有这两行是不敢跑的, 300行 0.58， 全量样本跑结果大概0.62
    # X = X[:300]
    # y = y[:300]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=2018)
    print(set(train_y), set(test_y))
    time_2 = time.time()
    print('read data cost ', time_2 - time_1, ' second', '\n')

    print('Start training')
    clf = LogisticRegressionClassifier()
    clf.fit(train_x, train_y)
    time_3 = time.time()
    print('training cost ', time_3 - time_2, ' seconds', '\n')

    print('-'*50)
    print('Start predicting')
    test_predict = clf.predict(test_x)
    time_4 = time.time()
    print('predicting cost ', time_4 - time_3, ' seconds', '\n')
    
    score = clf.score(test_x, test_y)
    print("The accruacy score is ", score)

    print('-'*40)
    score = clf.score(train_x, train_y)
    print("The accruacy score is ", score)

