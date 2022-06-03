from model.KDtree import KDtree
from model.DataPrecessPosUn import my_data
import numpy as np
import pandas as pd
from math import sqrt
da=my_data()
pa=my_data()
def nearest(tree, point, k):
    L = []
    def dis(x, p):
        if len(L) < k:
            d = sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(x, p)))
            L.append([x, d])
            return
        else:
            d = sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(x, p)))
            L.sort(key=lambda a: a[1])
            if (L[-1][1] > d):
                L.pop()
                L.append([x, d])
            return

    def travel(kd_node):
        if kd_node is None:
            return
        s = kd_node.split
        if kd_node.data[s] > point[s]:
            nearnode = kd_node.left
            furthnode = kd_node.right
        else:
            nearnode = kd_node.right
            furthnode = kd_node.left
        travel(nearnode)
        dis(kd_node.data, point)
        dis1 = abs(kd_node.data[s] - point[s])
        dis2 = sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(kd_node.data, point)))
        if len(L) < k or dis1 < dis2:
            travel(furthnode)
        else:
            return
    travel(tree.root)
    return L
def find_cores():
    test = np.zeros((len(da), len(da)))
    j, a, b, i, q, num, w, e, c, k = 0, 0, 0, 0, 0, 0, 0, 0,0,1
    knn = []
    f = []
    rnn = []
    while 1:
        while j < len(da):
            ii = 0
            kd = KDtree(pa)
            temp = da[j]
            knn = nearest(kd, temp, k + 1)
            while a < k + 1:
                if knn[a][1] == 0:
                    del knn[a]
                    break
                a = a + 1
            a = 0
            while c < len(knn):
                while b < len(da):
                    if knn[c][0] == da[b]:
                        test[j][b] = 1
                    b = b + 1
                b = 0
                c = c + 1
            c = 0
            j = j + 1
        j = 0
        rnn = test.sum(axis=0)
        num = 0
        while w < len(da):
            if rnn[w] == 0:
                num = num + 1
            w = w + 1
        w = 0
        if num == 0 or e == num or k == 8:
            break
        e = num
        k = k + 1
    a_mean = np.mean(rnn)
    a_std = np.std(rnn, ddof=1)
    t = a_mean - (-0.2 * a_std)
    P_C = []
    i = 0
    while i < len(da):
        if rnn[i] > t:
            P_C.append(i)
        i = i + 1
    return P_C

