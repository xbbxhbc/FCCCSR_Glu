from model.DataProcessTest import getTest
from model.DataPrecessPosUn import my_data,my_un_data
import numpy as np
from sklearn.metrics import matthews_corrcoef
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
np.random.seed(23)
import time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
def metrics(name,y_test,pre_lable,epoch_mins, epoch_secs):
    TN, FP, FN, TP = confusion_matrix(y_test, pre_lable).ravel()
    SN = recall_score(y_test, pre_lable)
    SP = TN / (TN + FP)
    MCC = matthews_corrcoef(y_test, pre_lable)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    print(f'Classifier: {name} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tSn: {SN *100:.2f}% | Sp: {SP * 100:.2f}%')
    print(f'\tACC: {ACC*100:.2f}%| MCC: {MCC :.4f}')
def pos_neg_1():
    start_time = time.time()
    da = my_data()
    unlabel = my_un_data()
    unlabel = unlabel[0:590]
    y_train = []
    X_train = []
    for te in da:
        X_train.append(te)
        y_train.append(1)
    i = 0
    while i < len(unlabel):
        X_train.append(unlabel[i])
        y_train.append(0)
        i = i + 1
    X_test, y_test = getTest()
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    clf = XGBClassifier(learning_rate=0.1826847, reg_alpha=1.32123653, reg_lambda=0.60855536, gamma=1.10605355,
                        max_depth=11, min_child_weight=1.68059523, subsample=0.17747384, colsample_bytree= 0.32800493)
    clf.fit(X_train, y_train)
    y_test = np.array(y_test)
    pre_lable = clf.predict(X_test)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    metrics("Pos:Neg(1:1)", y_test, pre_lable, epoch_mins, epoch_secs)
def pos_neg_2():
    start_time = time.time()
    da=my_data()
    unlabel=my_un_data()
    unlabel=unlabel[0:1180]
    y_train = []
    X_train = []
    for te in da:
        X_train.append(te)
        y_train.append(1)
    i = 0
    while i < len(unlabel):
        X_train.append(unlabel[i])
        y_train.append(0)
        i = i + 1
    X_test, y_test = getTest()
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    clf = XGBClassifier(learning_rate=0.20988609, reg_alpha=0.31732482, reg_lambda=0.53440738, gamma=0,
                        max_depth=13, min_child_weight= 9.87379187, subsample=0.32598439, colsample_bytree=0.56205341)
    clf.fit(X_train, y_train)
    y_test = np.array(y_test)
    pre_lable = clf.predict(X_test)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    metrics("Pos:Neg(1:2)", y_test, pre_lable,epoch_mins, epoch_secs)
def RandomSelect():
    pos_neg_1()
    pos_neg_2()