import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',
    "font.size": 8,
}
rcParams.update(config)
def getPosPro(temp):
    temp=temp.tolist()
    re=[]
    te=0
    while(te<len(temp)):
        re.append(temp[te][1])
        te=te+1
    return re
def ROC(y_true,XGBOOST,SVM,GBDT,RF,adaboost,DNN,cnn_y):
    y_true=y_true.tolist()
    XGBOOST=getPosPro(XGBOOST)
    SVM=getPosPro(SVM)
    GBDT= getPosPro(GBDT)
    RF=getPosPro(RF)
    adaboost=getPosPro(adaboost)
    fpr_1, tpr_1, thresholds_1 = metrics.roc_curve(y_true, adaboost, pos_label=1)
    fpr_2, tpr_2, thresholds_2 = metrics.roc_curve(y_true, GBDT, pos_label=1)
    fpr_3, tpr_3, thresholds_3 = metrics.roc_curve(y_true, RF, pos_label=1)
    fpr_4, tpr_4, thresholds_4 = metrics.roc_curve(y_true, SVM, pos_label=1)
    fpr_5, tpr_5, thresholds_5 = metrics.roc_curve(y_true, XGBOOST, pos_label=1)
    fpr_6, tpr_6, thresholds_6 = metrics.roc_curve(cnn_y,DNN,pos_label=1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('ROC curve', fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr_1, tpr_1, linestyle='--', c='#AFEEEE', linewidth=2)
    plt.plot(fpr_2, tpr_2, linestyle='--', c='#9932CC', linewidth=2)
    plt.plot(fpr_3, tpr_3, linestyle='--', c='#98FB98', linewidth=2)
    plt.plot(fpr_4, tpr_4, linestyle='--', c='#A9A9A9', linewidth=2)
    plt.plot(fpr_6, tpr_6, linestyle='--', c='#FFA500', linewidth=2)
    plt.plot(fpr_5, tpr_5, linestyle='--', c='#FF4500', linewidth=2)
    label = ['AdaBoost(AUC=0.7592)', 'GBDT(AUC=0.8573)', 'RF(AUC=0.8501)', 'SVM(AUC=0.8084)', 'CNN_DNN(AUC=0.8257)',
             'XGBoost(AUC=0.9077)']
    plt.plot([0, 1], [0, 1], color='#000000', linestyle='--')
    plt.legend(label, loc='lower right', fontsize=8)
    plt.show()
    # AUC_1 = auc(fpr_1, tpr_1)
    # AUC_2 = auc(fpr_2, tpr_2)
    # AUC_3 = auc(fpr_3, tpr_3)
    # AUC_4 = auc(fpr_4, tpr_4)
    # AUC_5 = auc(fpr_5, tpr_5)
    # AUC_6 = auc(fpr_6, tpr_6)

import numpy as np
from sklearn.metrics import precision_recall_curve
def PR(y_true,XGBOOST,SVM,GBDT,RF,adaboost,DNN,cnn_y):
    y_true=y_true.tolist()
    XGBOOST=getPosPro(XGBOOST)
    SVM=getPosPro(SVM)
    GBDT= getPosPro(GBDT)
    RF=getPosPro(RF)
    adaboost=getPosPro(adaboost)
    pre_1, re_1, thresholds = precision_recall_curve(y_true, adaboost, pos_label=1)
    pre_2, re_2, thresholds = precision_recall_curve(y_true, GBDT, pos_label=1)
    pre_3, re_3, thresholds = precision_recall_curve(y_true, RF, pos_label=1)
    pre_4, re_4, thresholds = precision_recall_curve(y_true, SVM, pos_label=1)
    pre_5, re_5, thresholds = precision_recall_curve(y_true, XGBOOST, pos_label=1)
    pre_6, re_6, thresholds = precision_recall_curve(cnn_y, DNN, pos_label=1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('PR curve', fontweight='bold')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot([0, 0.5], [0, 1], color='green', linestyle='--',linewidth=2)
    plt.plot(re_1, pre_1, linestyle='--', c='#AFEEEE', linewidth=2)
    plt.plot(re_2, pre_2, linestyle='--', c='#9932CC', linewidth=2)
    plt.plot(re_3, pre_3, linestyle='--', c='#98FB98', linewidth=2)
    plt.plot(re_4, pre_4, linestyle='--', c='#A9A9A9', linewidth=2)
    plt.plot(re_6,pre_6,linestyle='--',c='#FFA500',linewidth=2)
    plt.plot(re_5, pre_5, linestyle='--', c='#FF4500', linewidth=2)
    label = ['AdaBoost','GBDT','RF','SVM','CNN_DNN','XGBoost']
    plt.legend(label, loc='lower left', fontsize=8)
    plt.show()