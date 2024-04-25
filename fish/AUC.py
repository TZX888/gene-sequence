
import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
feature=np.load(file="/home/u2208283040/tzx/cscwd/feature_11.npy",allow_pickle=True)
feature=list(map(lambda x:x.upper(),feature))
X_train=np.load(file="/home/u2208283040/tzx/cscwd/k_11/data/X_train.npy")
y_train=np.load(file="/home/u2208283040/tzx/cscwd/k_11/data/y_train.npy")
X_test=np.load(file="/home/u2208283040/tzx/cscwd/k_11/data/X_test.npy")
y_test=np.load(file="/home/u2208283040/tzx/cscwd/k_11/data/y_test.npy")
d=[398369,3607439,15194,1307846,2678060,3412624,1995873,2095019,1232157,1119432,3142758,211199,16477,1033933,3482068,1036984,3989743,2640222,49406,193]

# d=[398369]
X=[]
Y=[]
column=[]
scoring = ['accuracy','precision_macro', 'recall_macro','f1_macro']#设置评分项
print(X_train.shape)
# X_17=np.load(file="/home/u2208283040/tzx/cscwd/k_11/AUC/X_1.npy")
# Y_18=np.load(file="/home/u2208283040/tzx/cscwd/k_11/AUC/Y_1.npy")
# print(X_17)
# print(Y_18)
for i in range(2,20):
    # X=[]
    # print(d.index(i))
    # for row in X_train:
    #         X.append([row[i]])
    # X=np.array(X)
    C=np.load(file="/home/u2208283040/tzx/cscwd/k_11/A_ROC/Y_"+str(i)+".npy")
    X=np.load(file="/home/u2208283040/tzx/cscwd/k_11/AUC/Y_"+str(i)+".npy")
    C=np.append(C,X,axis=1)
    print(C.shape)
    np.save(arr=C,file="/home/u2208283040/tzx/cscwd/k_11/A_ROC/Y_"+str(i+1)+".npy")
    # np.random.seed(12)
    # np.random.shuffle(X)
    # np.random.seed(12)
    # np.random.shuffle(y_train)
    # Y=[]
    # for row1 in X_test:
    #         Y.append([row1[i]])

    # Y=np.array(Y)
    # Y=np.load(file="/home/u2208283040/tzx/cscwd/k_11/AUC/Y_0.npy")
    # Y_1=np.load(file="/home/u2208283040/tzx/cscwd/k_11/AUC/Y_1.npy")
    

    # C=np.append(Y,Y_1,axis=1)
    # print(C.shape)
    # np.save(arr=C,file="/home/u2208283040/tzx/cscwd/k_11/A_ROC/Y_2.npy")
    # print(X_1.shape)
    # print(Y_1.shape)
    # print(np.append(X_1,X_1,axis=1))
#============================Random_Forest=========================


#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     print(X.shape)
# # # # # #============================Random_Forest=========================
#     model.fit(X, y_train)
# from sklearn.metrics import roc_auc_score
# preds=model.predict_proba(Y)
# print('AUC socre:',roc_auc_score(y_test, preds[:,-1])*100)
# model=joblib.load('/home/u2208283040/tzx/cscwd/k_11/model/Random_Forest.dat')
# print("模型保存成功")
# # y_pred=np.load(file="/home/u2208283040/tzx/cscwd/k_11/Random_Forest/y_pred.npy")
# # print("预测保存成功")
# # model=joblib.load('/home/u2208283040/tzx/cscwd/k_11/model/xgboot.dat')
# # print(y_pred)
# # 计算准确率
# pred=model.predict(X_test)
# preds=model.predict_proba(X_test)
# from sklearn.metrics import f1_score,classification_report,confusion_matrix
# # 模型评估指标
# from sklearn.metrics import precision_score,recall_score,roc_auc_score,roc_curve

# print("混淆矩阵：\n",confusion_matrix(y_test,pred))

# # (9)打印精确率、召回率、F1分数和AUC值、画出ROC曲线。（5分）
# ## accuracy
# from sklearn.metrics import accuracy_score

# print('ACC:',accuracy_score(y_test, pred))
# from sklearn import metrics

# print('Precision',metrics.precision_score(y_test, pred))
# from sklearn import metrics

# print('Recall',metrics.recall_score(y_test, pred))
# from sklearn import metrics

# print('F1-score:',metrics.f1_score(y_test, pred))
# import numpy as np
# from sklearn.metrics import roc_auc_score

# print('AUC socre:',roc_auc_score(y_test, preds[:,-1])*100)

