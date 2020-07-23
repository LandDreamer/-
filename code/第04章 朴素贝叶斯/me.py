import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 

from collections import Counter
import math


# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    # print(data)
    return data[:,:-1], data[:,-1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf1 = GaussianNB()
clf1.fit(X_train, y_train)
s1 = clf1.score(X_test, y_test)
print(s1)

pre1 = clf1.predict([[4.4,  3.2,  1.3,  0.2]])
print(pre1)


class Naive_Bayes():
    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    def stdev(self, X):
        avg = self.mean(X)
        return np.sqrt(sum([(x-avg)**2 for x in X]) / float(len(X)))

    def probability(self, x, mean, stdev, method='GaussianNB'):
        if method == 'GaussianNB':
            res = np.exp(-np.power(x-mean,2) / (2*stdev**2)) \
                    / np.sqrt(2*np.pi*stdev**2) 
        else:
            res = 0
        return res

    def summarize(self, X):
        res = [(self.mean(x), self.stdev(x)) for x in zip(*X)]
        return res

    def fit(self, X, Y):
        labels = list(set(Y))
        dataset = {label:[] for label in labels}

        for x,y in zip(X,Y):
            dataset[y].append(x)

        self.model = {
            label: (len(data)/len(Y), self.summarize(data))
            for label, data in dataset.items()
        }

        print('train done')
        return 
    
    def predict(self, input):
        labels = []
        for X in input:
            pro = {}
            for label, wei in self.model.items():
                pro[label] = wei[0]
                for i in range(len(wei[1])):
                    val = wei[1][i]
                    pro[label] *= self.probability(X[i], val[0], val[1])
        
            res = sorted(pro.items(), key=lambda x: x[-1])
            label = res[-1][0]
            labels.append(label)
        return labels

    def score(self, X, Y):
        res = self.predict(X)
        s = 0.
        for i in range(len(X)):
            if res[i] == Y[i]:
                s += 1
        return s / len(X)


print('-'*20)
print('my implement')

clf2 = Naive_Bayes()
clf2.fit(X_train, y_train)
s2 = clf2.score(X_test, y_test)
print(s2)

pre2 = clf2.predict([[4.4,  3.2,  1.3,  0.2]])
print(pre2)



