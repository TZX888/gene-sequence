import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
#matplotlib inline  

Gfish_dna = pd.read_csv('/home/u2208283040/tzx/LSTM/data_x/Gfish40.csv')
Hfish_dna = pd.read_csv('/home/u2208283040/tzx/LSTM/data_x/Hfish40.csv')

# Hfish_dna = pd.read_csv('test.csv')
# Gfish_dna = pd.read_csv('test1.csv')
# print(Hfish_dna.head())
# print(Gfish_dna.head())
print("读数据结束")
def Kmers_funct(seq, size=19):
    return [seq[x:x+size] for x in range(len(seq) - size + 1)]

Hfish_dna['data'] = Hfish_dna.apply(lambda x: Kmers_funct(x['data']), axis=1)
print("Hfish_dna结束")
# print(Hfish_dna.head())
Gfish_dna['data'] = Gfish_dna.apply(lambda x: Kmers_funct(x['data']), axis=1)
print("Gfish_dna结束")
# print(Gfish_dna.head())
fish=pd.concat([Hfish_dna,Gfish_dna],axis=0)
print(fish.head())

fish_texts = list(fish['data'])
for item in tqdm(range(len(fish_texts))):
    fish_texts[item] = ' '.join(fish_texts[item])
y_h = fish.iloc[:, 1].values      
print(y_h)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(fish_texts)
print(X.shape)
print("保存")
# np.save('/home/u2208283040/tzx/LSTM/dataVS/Kmers.npy',X)
# np.save('/home/u2208283040/tzx/LSTM/dataVS/Kmers_y.npy',y_h)
# X=np.load('/home/u2208283040/tzx/LSTM/dataVS/Kmers.npy',allow_pickle=True)
# y_h=np.load('/home/u2208283040/tzx/LSTM/dataVS/Kmers_y.npy',allow_pickle=True)
# fish['lable'].value_counts().sort_index().plot.bar(title="fish")
# plt.tight_layout()
# plt.show()
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
# smote过采样
smote = SMOTE(random_state=42,n_jobs=-1)
x_new, y_new = smote.fit_resample(X, y_h)
# print('smote后正样本数：', np.sum(y_new == 1), 'smote后负样本数：', np.sum(y_new == 0), 'smote后总数：', len(x_new))

print("训练")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_new, y_new, test_size = 0.30, random_state=42)
np.save(file="/home/u2208283040/tzx/LSTM/y_test_Kmer.npy",arr=y_test)
from collections import Counter
print("训练集各个标签的出现次数",Counter(y_train))
print("测试集各个标签的出现次数",Counter(y_test))
print("导入结束")
print(y_test)
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# #============================lightgbm=========================
# from lightgbm.sklearn import LGBMClassifier
# ## 定义 LightGBM 模型 
# lgbm= LGBMClassifier()
# X_train = X_train.astype(np.float32)
# X_test=X_test.astype(np.float32)
# lgbm.fit(X_train, y_train)
# y_pred = lgbm.predict(X_test)
# joblib.dump(lgbm,'/home/u2208283040/tzx/LSTM/LightGBM/LightGBM_Kmer.dat')
# print("模型保存成功")
# np.save(file="/home/u2208283040/tzx/LSTM/LightGBM/y_pred_Kmer.npy",arr=y_pred)
# print("预测保存成功")
# # 混淆矩阵
# cm = confusion_matrix(y_test, y_pred)
# title = 'Confusion matrix for LightGBM classifier'
# disp = ConfusionMatrixDisplay.from_estimator(lgbm,
#                                              X_test,
#                                              y_test,
#                                              cmap=plt.cm.Blues,
#                                              normalize=None,
#                                             )
# disp.ax_.set_title(title)
# plt.savefig('/home/u2208283040/tzx/LSTM/LightGBM/LightGBM_Kmer.eps')
# plt.savefig('/home/u2208283040/tzx/LSTM/LightGBM/LightGBM_Kmer.png')
# plt.show()
# print("混淆矩阵保存成功")
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# print("==================================ligbm_Kmer======================================")
# print("Confusion matrix on fish genes\n")
# print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
# def get_metrics(y_test, y_predicted):
#     accuracy = accuracy_score(y_test, y_predicted)
#     precision = precision_score(y_test, y_predicted, average='weighted')
#     recall = recall_score(y_test, y_predicted, average='weighted')
#     f1 = f1_score(y_test, y_predicted, average='weighted')
#     return accuracy, precision, recall, f1
# accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
# print("accuracy = %.4f \nprecision = %.4f \nrecall = %.4f \nf1 = %.4f" % (accuracy, precision, recall, f1))
# print("==================================ligbm_Kmer======================================")
# # #============================lightgbm=========================


# #============================rm=========================
# from sklearn.ensemble import RandomForestClassifier

# rm = RandomForestClassifier(n_estimators=100, random_state=42,class_weight="balanced")
# rm.fit(X_train, y_train)
# y_pred = rm.predict(X_test)
# joblib.dump(rm,'/home/u2208283040/tzx/LSTM/rm/rm_Kmer.dat')
# print("模型保存成功")
# np.save(file="/home/u2208283040/tzx/LSTM/rm/y_pred_Kmer.npy",arr=y_pred)
# print("预测保存成功")
# # 混淆矩阵
# cm = confusion_matrix(y_test, y_pred)
# title = 'Confusion matrix for Random_Forest classifier'
# disp = ConfusionMatrixDisplay.from_estimator(rm,
#                                              X_test,
#                                              y_test,
#                                              cmap=plt.cm.Blues,
#                                              normalize=None,
#                                             )
# disp.ax_.set_title(title)
# plt.savefig('/home/u2208283040/tzx/LSTM/rm/rm_Kmer.eps')
# plt.savefig('/home/u2208283040/tzx/LSTM/rm/rm_Kmer.png')
# plt.show()
# print("混淆矩阵保存成功")
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# print("==================================rm_Kmer======================================")
# print("Confusion matrix on fish genes\n")
# print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
# def get_metrics(y_test, y_predicted):
#     accuracy = accuracy_score(y_test, y_predicted)
#     precision = precision_score(y_test, y_predicted, average='weighted')
#     recall = recall_score(y_test, y_predicted, average='weighted')
#     f1 = f1_score(y_test, y_predicted, average='weighted')
#     return accuracy, precision, recall, f1
# accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
# print("accuracy = %.4f \nprecision = %.4f \nrecall = %.4f \nf1 = %.4f" % (accuracy, precision, recall, f1))
# print("==================================rm_Kmer======================================")
# # #============================rm=========================

# #============================DecisionTree=========================
# from sklearn.tree import DecisionTreeClassifier

# DecisionTree = DecisionTreeClassifier(random_state=42,class_weight="balanced")
# DecisionTree.fit(X_train, y_train)
# y_pred = DecisionTree.predict(X_test)
# joblib.dump(DecisionTree,'/home/u2208283040/tzx/LSTM/DecisionTree/DecisionTree_Kmer.dat')
# print("模型保存成功")
# np.save(file="/home/u2208283040/tzx/LSTM/DecisionTree/y_pred_Kmer.npy",arr=y_pred)
# print("预测保存成功")
# # 混淆矩阵
# cm = confusion_matrix(y_test, y_pred)
# title = 'Confusion matrix for DecisionTree classifier'
# disp = ConfusionMatrixDisplay.from_estimator(DecisionTree,
#                                              X_test,
#                                              y_test,
#                                              cmap=plt.cm.Blues,
#                                              normalize=None,
#                                             )
# disp.ax_.set_title(title)
# plt.savefig('/home/u2208283040/tzx/LSTM/DecisionTree/DecisionTree_Kmer.eps')
# plt.savefig('/home/u2208283040/tzx/LSTM/DecisionTree/DecisionTree_Kmer.png')
# plt.show()
# print("混淆矩阵保存成功")
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# print("==================================DecisionTree_Kmer======================================")
# print("Confusion matrix on fish genes\n")
# print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
# def get_metrics(y_test, y_predicted):
#     accuracy = accuracy_score(y_test, y_predicted)
#     precision = precision_score(y_test, y_predicted, average='weighted')
#     recall = recall_score(y_test, y_predicted, average='weighted')
#     f1 = f1_score(y_test, y_predicted, average='weighted')
#     return accuracy, precision, recall, f1
# accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
# print("accuracy = %.4f \nprecision = %.4f \nrecall = %.4f \nf1 = %.4f" % (accuracy, precision, recall, f1))
# print("==================================DecisionTree_Kmer======================================")
# # #============================DecisionTree=========================

#============================xgboost=========================
import xgboost as xgb
xgboost = xgb.XGBClassifier()
xgboost.fit(X_train, y_train)
y_pred = xgboost.predict(X_test)
joblib.dump(xgboost,'/home/u2208283040/tzx/LSTM/xgboost/xgboost_Kmer_19.dat')
print("模型保存成功")
np.save(file="/home/u2208283040/tzx/LSTM/xgboost/y_pred_Kmer_19.npy",arr=y_pred)
print("预测保存成功")
# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
title = 'Confusion matrix for xgboost classifier'
disp = ConfusionMatrixDisplay.from_estimator(xgboost,
                                             X_test,
                                             y_test,
                                             cmap=plt.cm.Blues,
                                             normalize=None,
                                            )
disp.ax_.set_title(title)
plt.savefig('/home/u2208283040/tzx/LSTM/xgboost/xgboost_Kmer_19.eps')
plt.savefig('/home/u2208283040/tzx/LSTM/xgboost/xgboost_Kmer_19.png')
plt.show()
print("混淆矩阵保存成功")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("==================================xgboost_Kmer======================================")
print("Confusion matrix on fish genes\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.4f \nprecision = %.4f \nrecall = %.4f \nf1 = %.4f" % (accuracy, precision, recall, f1))
print("==================================xgboost_Kmer======================================")
# #============================xgboost=========================
