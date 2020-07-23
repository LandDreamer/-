import pickle
import numpy as np
import pandas as pd
import math
from math import log
from collections import Counter
from Data import create_data

# 熵
def calc_ent(datasets):
    data_len = len(datasets)
    data_count = {}
    for i in range(data_len):
        label = datasets[i][-1]
        if label not in data_count:
            data_count[label] = 1.
        else:
            data_count[label] += 1

    res = -sum([x/data_len * log(x/data_len,2) for x in data_count.values()])
    return res 

# 经验条件熵
def calc_cond_ent(datasets, axis=0):
    data_len = len(datasets)
    feature_set = {}
    for i in range(data_len):
        feature = datasets[i][axis]
        if feature not in feature_set:
            feature_set[feature] = [datasets[i]]
        else:
            feature_set[feature].append(datasets[i])
    res = sum(
        [float(len(x))/data_len * calc_ent(x) for x in feature_set.values()]
    )
    return res

# 信息增益
def info_gain(ent, cond_ent):
    return ent-cond_ent

def info_gain_train(datasets, features):
    count = len(datasets[0])-1
    ent = calc_ent(datasets)
    result = []
    print("count is %d"%count)
    for i in range(count):
        cond_ent = calc_cond_ent(datasets, axis=i)
        info = info_gain(ent, cond_ent)
        result.append((info, i, features[i]))
        # print("%dth feature %s info gain is %f"%(i, features[i], info))

    best = sorted(result, key=lambda x: x[0])[-1]
    print("best is %f %dth %s" %best)
    # print(best)
    return best


if __name__=='__main__':
    datasets, features = create_data()
    info_gain_train(datasets, features)




