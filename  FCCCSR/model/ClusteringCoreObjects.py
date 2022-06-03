from model.KDtree import KDtree
from model.DataPrecessPosUn import my_data
from model.FindCoreObjects import find_cores
from math import sqrt
def cluster_cores():
    import numpy as np
    data = my_data()
    D1 = find_cores()
    da = []
    pa = []
    for te in D1:
        da.append(data[te])
        pa.append(data[te])
    test = np.zeros((len(da), len(da)))
    j, a, b, i, q, num, w, e, c, k,para= 0, 0, 0, 0, 0, 0, 0, 0,0, 1,4
    knn = []
    f = []
    rnn = []
    te_num = []
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
        if num == 0 or e == num or k ==para:
            break
        e = num
        te_num.append(num)
        k = k + 1
    te = []
    a1 = 0
    a2 = 0
    temp_mnn = []
    while a1 < len(D1):
        while a2 < len(D1):
            if test[a1][a2] == 1:
                if test[a1][a2] == test[a2][a1]:
                    te.append(a2)
            a2 = a2 + 1
        temp_mnn.append(te)
        te = []
        a2 = 0
        a1 = a1 + 1
    i = 0
    mnn_te = []
    mnn = []
    while i < len(temp_mnn):
        j = 0
        if len(temp_mnn[i]) == 0:
            mnn.append([])
        if len(temp_mnn[i]) != 0:
            while j < len(temp_mnn[i]):
                mnn_te.append(D1[temp_mnn[i][j]])
                j = j + 1
            mnn.append(mnn_te)
            mnn_te = []
        i = i + 1
    lable = [0 for index in range(len(D1))]
    m1 = 0
    m2 = 0
    la1 = 0
    la2 = 0
    tttemp = 0
    while tttemp < len(D1):
        if len(mnn[tttemp]) == 0:
            lable[tttemp] = -1
        tttemp = tttemp + 1
    while m1 < len(D1):
        np = []
        if lable[m1] == 0 and lable[m1] != -1:
            la = max(lable) + 1
            np.append(D1[m1])
            m2 = 0
            while m2 < len(mnn[m1]):
                np.append(mnn[m1][m2])
                m2 = m2 + 1
            m2 = 0
            lable[m1] = la
            while m2 < len(np):
                m3 = 0
                while m3 < len(D1):
                    if np[m2] == D1[m3]:
                        la1 = m3
                        break
                    m3 = m3 + 1
                if lable[la1] == 0 and lable[la1] != -1:
                    lable[la1] = la
                    tempnp = []
                    m4 = 0
                    while m4 < len(mnn[la1]):
                        tempnp.append(mnn[la1][m4])
                        m4 = m4 + 1
                    m4 = 0
                    while m4 < len(tempnp):
                        if D1[m1] == tempnp[m4]:
                            del tempnp[m4]
                        m4 = m4 + 1
                    m4 = 0
                    while m4 < len(tempnp):
                        if tempnp[m4] not in np:
                            m5 = 0
                            while m5 < len(D1):
                                if tempnp[m4] == D1[m5]:
                                    la2 = m5
                                    break
                                m5 = m5 + 1
                            if lable[la2] == 0 and lable[la2] != -1:
                                np.append(tempnp[m4])
                        m4 = m4 + 1
                m2 = m2 + 1
        m1 = m1 + 1
    i = 0
    a_1 = 0
    a_2 = 0
    a_3 = 0
    a_4 = 0
    a_5 = 0
    a_6 = 0
    a_7 = 0
    while i < len(lable):
        if lable[i] == -1:
            a_1 = a_1 + 1
        if lable[i] == 1:
            a_2 = a_2 + 1
        if lable[i] == 2:
            a_3 = a_3 + 1
        if lable[i] == 3:
            a_4 = a_4 + 1
        if lable[i] == 4:
            a_5 = a_5 + 1
        if lable[i] == 5:
            a_6 = a_6 + 1
        if lable[i] == 6:
            a_7 = a_7 + 1
        i = i + 1
    return lable

