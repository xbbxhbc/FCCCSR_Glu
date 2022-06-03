from model.ClusteringCoreObjects import cluster_cores
from model.KDtree import KDtree
from model.DataPrecessPosUn import my_data, my_un_data
from model.FindCoreObjects import find_cores
from math import sqrt
def select_reliable_negative():
    import numpy as np
    D1 = find_cores()
    lable = cluster_cores()
    c_1 = []
    c_2 = []
    c_3 = []
    c_4 = []
    c_5 = []
    c_6 = []
    i=0
    while i < len(lable):
        if lable[i] == 1:
            c_1.append(D1[i])
        if lable[i] == 2:
            c_2.append(D1[i])
        if lable[i] == 3:
            c_3.append(D1[i])
        if lable[i] == 4:
            c_4.append(D1[i])
        if lable[i] == 5:
            c_5.append(D1[i])
        if lable[i] == 6:
            c_6.append(D1[i])
        i = i + 1
    da = my_data()

    def get_cluster_center(temp):
        b1 = 0
        c1 = [0 for index in range(len(da[0]))]
        while b1 < len(temp):
            c1 = np.sum([da[temp[b1]], c1], axis=0).tolist()
            b1 = b1 + 1
        c_mean = np.divide(c1, len(temp))
        return c_mean
    c_1_mean = get_cluster_center(c_1)
    c_2_mean = get_cluster_center(c_2)
    c_3_mean = get_cluster_center(c_3)
    c_4_mean = get_cluster_center(c_4)
    c_5_mean = get_cluster_center(c_5)
    c_6_mean = get_cluster_center(c_6)
    def get_cluster_r(c_mean, cluster, r):
        i = 0
        dis = []
        while i < len(cluster):
            dis.append(np.sqrt(np.sum(np.square(c_mean - da[cluster[i]]))))
            i = i + 1
        c_r = max(dis)
        c_r = c_r * r
        return c_r
    t = 1.12
    c_1_r = get_cluster_r(c_1_mean, c_1, t)
    c_2_r = get_cluster_r(c_2_mean, c_2, t)
    c_3_r = get_cluster_r(c_3_mean, c_3, t)
    c_4_r = get_cluster_r(c_4_mean, c_4, t)
    c_5_r = get_cluster_r(c_5_mean, c_5, t)
    c_6_r = get_cluster_r(c_6_mean, c_6, t)
    unlabel = []
    unlabel = my_un_data()
    i = 0
    unlabel_lable = []
    unlabel_lable = [-1 for index in range(len(unlabel))]
    a = 0
    f_index = []
    while i < len(unlabel):
        c_1_temp = np.sqrt(np.sum(np.square(c_1_mean - unlabel[i])))
        if c_1_temp < c_1_r:
            a = a + 1
        c_2_temp = np.sqrt(np.sum(np.square(c_2_mean - unlabel[i])))
        if c_2_temp < c_2_r:
            a = a + 1
        c_3_temp = np.sqrt(np.sum(np.square(c_3_mean - unlabel[i])))
        if c_3_temp < c_3_r:
            a = a + 1
        c_4_temp = np.sqrt(np.sum(np.square(c_4_mean - unlabel[i])))
        if c_4_temp < c_4_r:
            a = a + 1
        c_5_temp = np.sqrt(np.sum(np.square(c_5_mean - unlabel[i])))
        if c_5_temp < c_5_r:
            a = a + 1
        c_6_temp = np.sqrt(np.sum(np.square(c_6_mean - unlabel[i])))
        if c_6_temp < c_6_r:
            a = a + 1
        if a > 0:
            unlabel_lable[i] = 1
        i = i + 1
        a = 0
    i = 0
    f_index = []
    while i < len(unlabel_lable):
        if unlabel_lable[i] == -1:
            f_index.append(i)
        i = i + 1
    f_da = []
    i = 0
    while i < len(f_index):
        f_da.append(unlabel[f_index[i]])
        i = i + 1
    return f_da











