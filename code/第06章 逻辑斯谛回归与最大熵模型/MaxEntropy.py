import numpy as np
import pandas as pd
import math
from copy import deepcopy

# fi(X,y) = 1 if and only if (x,y) id is i
# and X has x feature
# eg: X = ['good', 'sum'] and x = 'good'  
class MaxEntropy:
    def __init__(self, eta=0.0001, iters = 1000):
        self.eta = eta
        self.iters = iters

        self.datasets = None
        self.N = None # batch_size
        self.n = None # num of (x,y)
        self.M = None # 

        self.Y = None
        self.EP_ = {}
        self.xy2id = {}
        self.id2xy = {}
        self.num_xy = {}

        self.w = []
        self.last_w = []

    def fit(self, datasets):
        self.load_datasets(datasets)
        self.train()

    def load_datasets(self, datasets):
        self.datasets = deepcopy(datasets)
        
        self.M = max([len(X)-1 for X in datasets])
        self.N = len(datasets)
        self.Y = set()
        for sample in datasets:
            y = sample[0]
            self.Y.add(y)
            X = sample[1:]
            for x in X:
                if (x, y) in self.num_xy:
                    self.num_xy[(x, y)] += 1.
                else:
                    self.num_xy[(x, y)] = 1.

        self.n = len(self.num_xy)
        self.w = [0.] * self.n
        self.last_w = self.w
        for i, xy in enumerate(self.num_xy):
            self.xy2id[xy] = i
            self.id2xy[i] = xy
            self.EP_[xy] = self.num_xy[xy] / self.N

    def train(self):
        for loop in range(1, self.iters):
            self.last_w = self.w[:]
            for id in range(self.n):
                x, y = self.id2xy[id]
                ep = self.calc_EP(id)
                # print('ep is ', ep)
                self.w[id] += math.log(self.EP_[(x, y)]/ep) / self.M
            if self.convergence():
                break
            if loop%10 == 0:
                print('iter ', loop, ' times')

    def calc_EP(self, id):
        # E_P(fi)
        x, y = self.id2xy[id]
        res = 0.
        for data in self.datasets:
            X = data[1:]
            if x not in X:
                continue
            # if x is in X 
            # then fi(x, y) = 1
            # so we can calculate P(y|x)
            res += self.calc_Pyx(X, y) 

        res /= self.N
        return res

    def calc_Pyx(self, X, y):
        # P(y|x)
        Z = self.Zx(X)
        s = 0.
        for x in X:
            if (x, y) not in self.num_xy:
                continue
            s += self.w[self.xy2id[(x, y)]]
        res = math.exp(s)
        return res

    def Zx(self, X):
        res = 0.
        for y in self.Y:
            s = 0.
            for x in X:
                if (x, y) not in self.num_xy:
                    continue
                s += self.w[self.xy2id[(x, y)]]     
            res += math.exp(s)
        return res         

    def convergence(self):   
        for last, now in zip(self.last_w, self.w):
            if abs(last - now) > self.eta:
                return False
        return True

    def predict(self, input):
        res = []
        for X in input:
            tmp = []
            for y in self.Y:
                tmp.append((y, self.calc_Pyx(X, y)))
            res.append(max(tmp, key=lambda x: x[-1])[0])
        return res

    def score(self, X, y):
        prediction = (self.predict(X))
        print(prediction)
        print(y)
        right = 0.
        for i in range(len(y)):
            if prediction[i] == y[i]:
                right += 1.
        return right / len(y)

def load_data():
    dataset = [['no', 'sunny', 'hot', 'high', 'FALSE'],
           ['no', 'sunny', 'hot', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'high', 'FALSE'],
           ['yes', 'rainy', 'mild', 'high', 'FALSE'],
           ['yes', 'rainy', 'cool', 'normal', 'FALSE'],
           ['no', 'rainy', 'cool', 'normal', 'TRUE'],
           ['yes', 'overcast', 'cool', 'normal', 'TRUE'],
           ['no', 'sunny', 'mild', 'high', 'FALSE'],
           ['yes', 'sunny', 'cool', 'normal', 'FALSE'],
           ['yes', 'rainy', 'mild', 'normal', 'FALSE'],
           ['yes', 'sunny', 'mild', 'normal', 'TRUE'],
           ['yes', 'overcast', 'mild', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'normal', 'FALSE'],
           ['no', 'rainy', 'mild', 'high', 'TRUE']]
    dataset = np.array(dataset)
    return dataset

dataset = load_data()
maxent = MaxEntropy()
x = [['overcast', 'mild', 'high', 'FALSE']]
a = [[1,2,3,4],[2,3,4,5]]
X = dataset[:, 1:]
y = dataset[:, 0]
maxent.fit(dataset)
score = maxent.score(X, y)
print(score)