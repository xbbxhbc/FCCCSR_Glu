from model.SelectReliableNegativeSamples import select_reliable_negative
from model.DataPrecessPosUn import my_data
def getTrain():
    f_da = select_reliable_negative()
    da = my_data()
    y_train = []
    X_train = []
    i = 0
    while i < len(da):
        X_train.append(da[i])
        y_train.append(1)
        i = i + 1
    i = 0
    while i < len(f_da):
        X_train.append(f_da[i])
        y_train.append(0)
        i = i + 1
    return X_train,y_train
