import pickle
import numpy as np
import pandas as pd

def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
    features = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']

    return datasets, features
    
if __name__=='__main__':
    datasets, features = create_data()
    train_df = pd.DataFrame(datasets, columns=features)
    # print(train_df)
    a = train_df.iloc[:,-1]
    # print(set(a))
    # print(a.value_counts())
    # print(train_df.loc[train_df[u'类别'] == u'是'])
    # y = dict(a.value_counts())
    # print(max(y, key=lambda x: x[-1])[0])
    # print(a)
    # print(a.iloc[0])
    b = [1,2,3,4]
    c = b.copy()
    b.pop(0)
    print(c)


