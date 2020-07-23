import numpy as np
import pandas as pd
import sklearn
from collections import namedtuple
import math

class KdNode(object):
    def __init__(self, node, split, left, right):
        self.node = node
        self.split = split
        self.left = left
        self.right = right

class KdTree(object):
    def __init__(self, dataset):
        k = len(dataset[0]) # weidu

        def Createnode(split, dataset):
            if not dataset:
                return None
            next_num = (split+1) % k
            dataset.sort(key = lambda x:x[split])
            pos = len(dataset) // 2
            d1 = dataset[:pos]
            d2 = dataset[pos+1:]

            return KdNode(
                dataset[pos],
                split,
                Createnode(next_num, d1),
                Createnode(next_num, d2)
            )

        self.root = Createnode(0, dataset)
        
def preorder(node):
    print(node.node)
    if node.left:
        preorder(node.left)
    if node.right:
        preorder(node.right)

data = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
kd = KdTree(data)
# preorder(kd.root)      

def cal_dis(x,y, method='normal'):
    if method == 'normal':
        dis = math.sqrt(sum((p1-p2)**2 for p1,p2 in zip(x,y)))
    elif method == 'mahalanobis':
        x = np.array(x)
        y = np.array(y)
        D = np.cov(x.T, y.T)
        invD = np.linalg.inv(D)
        dis = (x-y).T.dot(invD.dot(x-y))
    return dis


result = namedtuple('result', ['point', 'dis'])
t = result([], [])

def find_Knear(dataset, point, k_num=1, method='normal'):
    kd = KdTree(dataset)
    k = len(point)
    
    def travel(kdnode, target, k_num=1, method='normal'):
        if kdnode is None:
            return
        node = kdnode.node
        print(node)
        split = kdnode.split

        if target[split] <= node[split]: 
            f_node = kdnode.left
            s_node = kdnode.right
        else:
            s_node = kdnode.left
            f_node = kdnode.right
        
        travel(f_node, target, k_num=k_num, method=method)
        dis = cal_dis(node, target, method=method)

        if len(t.point) < k_num:
            t.point.append(node)
            t.dis.append(dis)
            max_dis = max(t.dis)
            if abs(node[split] - target[split]) < max_dis:
                travel(s_node, target, k_num=k_num, method=method)
            return 

        max_dis = max(t.dis)
        if max_dis > dis:
            print(len(t.dis))
            for i in range(len(t.dis)):
                if t.dis[i] == max_dis:
                    print('change')
                    print(t.point)
                    t.point[i] = node
                    t.dis[i] = dis
                break

        max_dis = max(t.dis)
        if abs(node[split] - target[split]) < max_dis:
            travel(s_node, target, k_num=k_num, method=method)

    travel(kd.root, point, k_num=k_num,method=method)


find_Knear(data, [3,4.5], k_num=2, method='mahalanobis')
print (t)

