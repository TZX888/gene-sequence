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
import warnings
warnings.filterwarnings("ignore")
Gfish_dna = pd.read_csv('/home/u2208283040/tzx/LSTM/data_x/Gfish40.csv')
Hfish_dna = pd.read_csv('/home/u2208283040/tzx/LSTM/data_x/Hfish40.csv')

# Hfish_dna = pd.read_csv('test.csv')
# Gfish_dna = pd.read_csv('test1.csv')
# print(Hfish_dna.head())
# print(Gfish_dna.head())
print("读数据结束")
fish=pd.concat([Hfish_dna,Gfish_dna],axis=0)
print(fish.shape[0])
# function to convert a DNA sequence string to a numpy array
# converts to lower case, changes any non 'acgt' characters to 'n'
import numpy as np
def integer_encode(seq):
    encoding = {'A': 1, 'T': 2, 'C': 3, 'G': 4,'N': 0}
    return [encoding[x] for x in seq]
res=[]
X=[]
length=400000
for seq in tqdm(fish["data"]):
    res=integer_encode(seq)
    if len(res) < length: #小于指定的最大长度则用0填充
        res.extend([0] * (length - len(res)))
    X.append(res)
y_h=[]
for i in fish['lable']:
    y_h.append(i)
print("数据集")
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
# smote过采样
smote = SMOTE(random_state=42,n_jobs=-1)
x_new, y_new = smote.fit_resample(X, y_h)
print('smote后正样本数：', np.sum(y_new == 1), 'smote后负样本数：', np.sum(y_new == 0), 'smote后总数：', len(x_new))
np.save(file="/home/u2208283040/tzx/LSTM/y_ture_One.npy",arr=y_new)
# 随机欠采样
# rus = RandomUnderSampler()
# x_new, y_new = rus.fit_resample(X, y_h)
# print('随机欠采样后正样本数：', np.sum(y_new == 1), '随机欠采样后负样本数：', np.sum(y_new == 0), '随机欠采样后总数：', len(x_new))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_new, y_new, test_size = 0.30, random_state=42)
# print("保存")
# import h5py
# h5f = h5py.File('/home/u2208283040/tzx/LSTM/dataVS/Onehot.h5', 'w')
# h5f.create_dataset('dataset_train_x', data=X_train)
# h5f.create_dataset('dataset_train_y', data=y_train)
# h5f.create_dataset('dataset_test_x', data=X_test)
# h5f.create_dataset('dataset_test_y', data=y_test)
# h5f.close()
print("训练")
# # h5f = h5py.File('aug1202data.h5','r')
# # train_x = h5f['dataset_train_x']
# # y_train = h5f['dataset_train_y']
# # h5f.close()

from collections import Counter
print("训练集各个标签的出现次数",Counter(y_train))
print("测试集各个标签的出现次数",Counter(y_test))
print("导入结束")
print(y_test)
# ### Multinomial Naive Bayes Classifier ###
# # The alpha parameter was determined by grid search previously
# from sklearn.naive_bayes import MultinomialNB
# classifier = MultinomialNB(alpha=0.1)
# classifier.fit(X_train, y_train)

# y_pred = classifier.predict(X_test)


# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
import h5py
# h5f = h5py.File('/home/u2208283040/tzx/LSTM/dataVS/Seqcode.h5','r')
# X_train = h5f['dataset_train_x']
# y_train = h5f['dataset_train_y']
# X_test = h5f['dataset_test_x']
# y_test = h5f['dataset_test_y']
# h5f.close()
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#============================lightgbm=========================
from lightgbm.sklearn import LGBMClassifier
## 定义 LightGBM 模型 
lgbm= LGBMClassifier(class_weight="balanced")
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
joblib.dump(lgbm,'/home/u2208283040/tzx/LSTM/LightGBM/LightGBM_One.dat')
print("模型保存成功")
np.save(file="/home/u2208283040/tzx/LSTM/LightGBM/y_pred_One.npy",arr=y_pred)
print("预测保存成功")
# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
title = 'Confusion matrix for LightGBM classifier'
disp = ConfusionMatrixDisplay.from_estimator(lgbm,
                                             X_test,
                                             y_test,
                                             cmap=plt.cm.Blues,
                                             normalize=None,
                                            )
disp.ax_.set_title(title)
plt.savefig('/home/u2208283040/tzx/LSTM/LightGBM/LightGBM_One.eps')
plt.savefig('/home/u2208283040/tzx/LSTM/LightGBM/LightGBM_One.png')
plt.show()
print("混淆矩阵保存成功")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("==================================ligbm_One======================================")
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
print("==================================ligbm_One======================================")
# #============================lightgbm=========================


#============================rm=========================
from sklearn.ensemble import RandomForestClassifier

rm = RandomForestClassifier(n_estimators=100, random_state=42,class_weight="balanced")
rm.fit(X_train, y_train)
y_pred = rm.predict(X_test)
joblib.dump(rm,'/home/u2208283040/tzx/LSTM/rm/rm_One.dat')
print("模型保存成功")
np.save(file="/home/u2208283040/tzx/LSTM/rm/y_pred_One.npy",arr=y_pred)
print("预测保存成功")
# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
title = 'Confusion matrix for Random_Forest classifier'
disp = ConfusionMatrixDisplay.from_estimator(rm,
                                             X_test,
                                             y_test,
                                             cmap=plt.cm.Blues,
                                             normalize=None,
                                            )
disp.ax_.set_title(title)
plt.savefig('/home/u2208283040/tzx/LSTM/rm/rm_One.eps')
plt.savefig('/home/u2208283040/tzx/LSTM/rm/rm_One.png')
plt.show()
print("混淆矩阵保存成功")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("==================================rm_One======================================")
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
print("==================================rm_One======================================")
# #============================rm=========================

#============================DecisionTree=========================
from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(random_state=42,class_weight="balanced")
DecisionTree.fit(X_train, y_train)
y_pred = DecisionTree.predict(X_test)
joblib.dump(DecisionTree,'/home/u2208283040/tzx/LSTM/DecisionTree/DecisionTree_One.dat')
print("模型保存成功")
np.save(file="/home/u2208283040/tzx/LSTM/DecisionTree/y_pred_One.npy",arr=y_pred)
print("预测保存成功")
# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
title = 'Confusion matrix for DecisionTree classifier'
disp = ConfusionMatrixDisplay.from_estimator(DecisionTree,
                                             X_test,
                                             y_test,
                                             cmap=plt.cm.Blues,
                                             normalize=None,
                                            )
disp.ax_.set_title(title)
plt.savefig('/home/u2208283040/tzx/LSTM/DecisionTree/DecisionTree_One.eps')
plt.savefig('/home/u2208283040/tzx/LSTM/DecisionTree/DecisionTree_One.png')
plt.show()
print("混淆矩阵保存成功")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("==================================DecisionTree_One======================================")
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
print("==================================DecisionTree_One======================================")
# #============================DecisionTree=========================

#============================xgboost=========================
import xgboost as xgb
xgboost = xgb.XGBClassifier()
xgboost.fit(X_train, y_train)
y_pred = xgboost.predict(X_test)
joblib.dump(xgboost,'/home/u2208283040/tzx/LSTM/xgboost/xgboost_One.dat')
print("模型保存成功")
np.save(file="/home/u2208283040/tzx/LSTM/xgboost/y_pred_One.npy",arr=y_pred)
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
plt.savefig('/home/u2208283040/tzx/LSTM/xgboost/xgboost_One.eps')
plt.savefig('/home/u2208283040/tzx/LSTM/xgboost/xgboost_One.png')
plt.show()
print("混淆矩阵保存成功")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("==================================xgboost_One======================================")
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
print("==================================xgboost_One======================================")
# #============================xgboost=========================






