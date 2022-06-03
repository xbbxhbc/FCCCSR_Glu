import pandas as pd
def getTest():
    fpath = "./data/Intest.txt"
    train = pd.read_csv(fpath)
    train = pd.read_csv(
        fpath,
        sep=' ',
        header=None,
    )
    ii = 0
    test_p = []
    while ii < 54:
        aa = train.loc[ii, 0:75].tolist()
        aa = list(map(float, aa))
        test_p.append(aa)
        ii += 1
    test_f = []
    while ii < len(train):
        aa = train.loc[ii, 0:75].tolist()
        aa = list(map(float, aa))
        test_f.append(aa)
        ii += 1
    y_test = []
    X_test = []
    i = 0
    while i < len(test_p):
        X_test.append(test_p[i])
        y_test.append(1)
        i = i + 1
    i = 0
    while i < len(test_f):
        X_test.append(test_f[i])
        y_test.append(0)
        i = i + 1
    return  X_test,y_test












