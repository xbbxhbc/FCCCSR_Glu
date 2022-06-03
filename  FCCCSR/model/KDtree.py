import numpy as np
import pandas as pd
from math import sqrt
class KDnode:
    def __init__(self, data, left, right, split):
        self.left = left
        self.right = right
        self.split = split
        self.data = data
class KDtree:
    def __init__(self, data):
        self.k = len(data[0])

        def CreatKD(split, data_set):
            if not data_set:
                return None
            data_set.sort(key=lambda x: x[split])
            flag = len(data_set) // 2
            new_split = (split + 1) % self.k
            return KDnode(data_set[flag], CreatKD(new_split, data_set[:flag]), CreatKD(new_split, data_set[flag + 1:]),
                          split)
        self.root = CreatKD(0, data)
