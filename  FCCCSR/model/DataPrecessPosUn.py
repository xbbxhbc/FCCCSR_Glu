import  pandas as pd
def my_data():
    pos_num = 590
    fpath = "./data/GI_76.txt"
    train = pd.read_csv(fpath)
    train = pd.read_csv(
        fpath,
        sep=' ',
        header=None,
    )
    ii = 0
    da = []
    while ii < pos_num:
        aa = train.loc[ii, 0:75].tolist()
        aa = list(map(float, aa))
        da.append(aa)
        ii += 1
    return da
def my_un_data():
    pos_num = 590
    fpath="./data/GI_76.txt"
    train = pd.read_csv(fpath)
    train = pd.read_csv(
        fpath,
        sep=' ',
        header=None,
    )
    ii = pos_num
    unlabel = []
    while ii <len(train):
        aa = train.loc[ii, 0:75].tolist()
        aa = list(map(float, aa))
        unlabel.append(aa)
        ii += 1
    return unlabel