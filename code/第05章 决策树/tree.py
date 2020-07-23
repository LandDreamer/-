import pickle
import numpy as np
import pandas as pd
import math
from math import log
from collections import Counter
from Data import create_data
from util import info_gain_train
import pprint
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, leave=False, label=None, feature_name=None, feature=None, info_gain_score=None):
        self.leave = leave
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree_node = {}
        self.info_gain_score = info_gain_score

    def __str__(self):
        return '{}'.format({
            'tree': self.tree_node,
            'feature_name': self.feature_name,
            'label': self.label
            #'info_gain_score': self.info_gain_score
        })

    def __repr__(self):
        return self.__str__()

    def add_node(self, val, node):
        self.tree_node[val] = node

    def predict(self, x):
        if self.leave is True:
            return self.label
        # print(self.feature)
        node = self.tree_node[x[self.feature]]
        x.pop(self.feature)
        return node.predict(x)

class DTree:
    def __init__(self, eta=0.2):
        self.eta = eta #阈值eta
        self.tree_root = {}

    def train(self, train_df):
        """
        input:数据集D(DataFrame格式)，特征集A，阈值eta
        output:决策树T
        """

        y = train_df.iloc[:,-1]
        features = train_df.columns[:-1]

        if len(y.value_counts()) == 1:
            return Node(leave=True, label=y.iloc[0])

        if len(features) == 0:
            return Node(leave=True, label=max(dict(y.value_counts()), key=lambda x: x[-1])[0])

        datasets = np.array(train_df)

        info_score, max_feature, max_feature_name = info_gain_train(datasets, features)

        if info_score < self.eta:
            return Node(leave=True, label=max(dict(y.value_counts()), key=lambda x: x[-1])[0])
            
        node = Node(
            feature_name=max_feature_name, 
            feature=max_feature,
            info_gain_score=info_score
        )
        
        label_list = set(train_df[max_feature_name])

        for label in label_list:
            sub_df = train_df.loc[train_df[max_feature_name] == label].drop([max_feature_name], axis=1)
            node.add_node(
                label,
                self.train(sub_df)
            )

        return node

    def fit(self, train_df):
        self.tree_root = self.train(train_df)
        return self.tree_root

    def predict(self, X_test):
        '''
        X_test list[list[str]]
        '''
        res = []
        for x in X_test:
            res.append(self.tree_root.predict(x))
        return res 

    def score(self, X_test, Y_test):
        res = self.predict(X_test)
        s = 0.
        for i in range(len(res)):
            if res[i] == Y_test[i]:
                s += 1
        return s / len(Y_test)


if __name__=='__main__':
    datasets, features = create_data()
    train_df = pd.DataFrame(datasets, columns=features)
#     dt = DTree()
#     tree = dt.fit(train_df)
#     print(tree)

# print(dt.predict([
#         ['老年', '否', '否', '一般'],
#         ['青年', '是', '否', '好'], 
#         ['青年', '是', '是', '一般']]
#     ))

    # iris = load_iris()
    # df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # df['label'] = iris.target
    # df.columns = [
    #     'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    # ]
    # # print(data)
    # train_df = df.iloc[:100]

    # train_df, test_df = train_test_split(train_df,test_size=0.3)
    dt = DTree()
    dt.fit(train_df)
    test = np.array(train_df)
    X_test = test[:,:-1]
    Y_test = test[:,-1]
    print(dt.score(X_test.tolist(),Y_test.tolist()))