from model.DataProcessTest import getTest
from model.DataProcessTrain import getTrain
import numpy as np
from sklearn.metrics import matthews_corrcoef
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import  AdaBoostClassifier
from model.ROC_PR import ROC,PR
from model.CNN_DNN import  cnn_dnn
np.random.seed(23)
import time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
def metrics(name, y_test, pre_lable, epoch_mins, epoch_secs):
    TN, FP, FN, TP = confusion_matrix(y_test, pre_lable).ravel()
    SN = recall_score(y_test, pre_lable)
    SP = TN / (TN + FP)
    MCC = matthews_corrcoef(y_test, pre_lable)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    print(f'Classifier: {name} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tSn: {SN * 100:.2f}% | Sp: {SP * 100:.2f}%')
    print(f'\tACC: {ACC * 100:.2f}%| MCC: {MCC :.4f}')
def FCCCSR_Main():
    start_time = time.time()
    X_train, y_train = getTrain()
    X_test, y_test = getTest()
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    clf = XGBClassifier(learning_rate=0.12582814, reg_alpha=0.50806786, reg_lambda=1.44791413, gamma=1.49697269,
                        max_depth=10, min_child_weight=1.5826088, subsample=0.79083351, colsample_bytree=0.99536393)
    clf.fit(X_train, y_train)
    y_test = np.array(y_test)
    pre_lable = clf.predict(X_test)
    XGBoost_pro=clf.predict_proba(X_test)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    metrics("XGBoost", y_test, pre_lable, epoch_mins, epoch_secs)
    # start_time = time.time()
    clf = SVC(kernel='linear', C=10, probability=True)
    clf.fit(X_train, y_train)
    pre_lable = clf.predict(X_test)
    SVM_pro = clf.predict_proba(X_test)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    metrics("SVM", y_test, pre_lable, epoch_mins, epoch_secs)
    # start_time = time.time()
    clf = GradientBoostingClassifier(n_estimators=220)
    clf.fit(X_train, y_train)
    pre_lable = clf.predict(X_test)
    GBDT_pro = clf.predict_proba(X_test)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    metrics("GBDT", y_test, pre_lable, epoch_mins, epoch_secs)
    # start_time = time.time()
    clf = RandomForestClassifier(n_estimators=212, criterion='gini', max_depth=None, min_samples_split=2,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                 max_leaf_nodes=None,
                                 min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=1, random_state=10,
                                 verbose=0, warm_start=False, class_weight=None)
    clf.fit(X_train, y_train)
    pre_lable = clf.predict(X_test)
    RF_pro = clf.predict_proba(X_test)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    metrics("RandomForest", y_test, pre_lable, epoch_mins, epoch_secs)
    # start_time = time.time()
    clf = AdaBoostClassifier(n_estimators=134, learning_rate=1.28153539, random_state=10)
    clf.fit(X_train, y_train)
    pre_lable = clf.predict(X_test)
    AdaBoost_pro = clf.predict_proba(X_test)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    metrics("AdaBoost", y_test, pre_lable, epoch_mins, epoch_secs)
    #---CNN_DNN---------------
    CNN_DNN_pro,cnn_y=cnn_dnn()
    ROC(y_test, XGBoost_pro,SVM_pro,GBDT_pro,RF_pro,AdaBoost_pro,CNN_DNN_pro,cnn_y)
    PR(y_test, XGBoost_pro,SVM_pro,GBDT_pro,RF_pro,AdaBoost_pro,CNN_DNN_pro,cnn_y)














