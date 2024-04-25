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
fish=pd.concat([Hfish_dna,Gfish_dna],axis=0)
print(fish.head())
print(fish.shape[0],fish.shape[1])

# function to convert a DNA sequence string to a numpy array
# converts to lower case, changes any non 'acgt' characters to 'n'
import numpy as np
import re
def string_to_array(my_string):
    my_string = my_string.lower()
    my_string = re.sub('[^acgt]', 'z', my_string)
    my_array = np.array(list(my_string))
    return my_array

# create a label encoder with 'acgtn' alphabet
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['a','c','g','t','z']))
def ordinal_encoder(my_array):
    integer_encoded = label_encoder.transform(my_array)
    float_encoded = integer_encoded.astype(float)
    float_encoded[float_encoded == 0] = 0.25 # A
    float_encoded[float_encoded == 1] = 0.50 # C
    float_encoded[float_encoded == 2] = 0.75 # G
    float_encoded[float_encoded == 3] = 1.00 # T
    float_encoded[float_encoded == 4] = 0.00 # anything else, z
    return list(float_encoded)
y_h=[]
for i in fish['lable']:
    y_h.append(i)
print(len(y_h))
X=[]
res=[]
for test_sequence in tqdm(fish['data']):
    if len(test_sequence) < 400000: #小于指定的最大长度则用0填充
        res=ordinal_encoder(string_to_array(test_sequence))
        res.extend([0] * (400000 - len(test_sequence)))
    # X=ordinal_encoder(string_to_array(test_sequence))
        X.extend([res])
print(len(X))
print("数据集")
# print(X)
from collections import Counter
print("各个标签的出现次数",Counter(y_h))
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
# smote过采样
smote = SMOTE(random_state=42,n_jobs=-1)
x_new, y_new = smote.fit_resample(X, y_h)
print('smote后正样本数：', np.sum(y_new == 1), 'smote后负样本数：', np.sum(y_new == 0), 'smote后总数：', len(x_new))

# 随机欠采样
# rus = RandomUnderSampler()
# x_new, y_new = rus.fit_resample(X, y_h)
# print('随机欠采样后正样本数：', np.sum(y_new == 1), '随机欠采样后负样本数：', np.sum(y_new == 0), '随机欠采样后总数：', len(x_new))

X_train, X_test, y_train, y_test = train_test_split(x_new, y_new, test_size = 0.30, random_state=42)
# print("保存")
# import h5py
# h5f = h5py.File('/home/u2208283040/tzx/LSTM/dataVS/Seqcode.h5', 'w')
# h5f.create_dataset('dataset_train_x', data=X_train)
# h5f.create_dataset('dataset_train_y', data=y_train)
# h5f.create_dataset('dataset_test_x', data=X_test)
# h5f.create_dataset('dataset_test_y', data=y_test)
# h5f.close()
# print("训练")
import h5py
# h5f = h5py.File('/home/u2208283040/tzx/LSTM/dataVS/Seqcode.h5','r')
# X_train = h5f['dataset_train_x']
# y_train = h5f['dataset_train_y']
# X_test = h5f['dataset_test_x']
# y_test = h5f['dataset_test_y']
# h5f.close()
from collections import Counter
print("训练集各个标签的出现次数",Counter(y_train))
print("测试集各个标签的出现次数",Counter(y_test))
print("导入结束")
print(y_test)
# np.random.seed(12)
# np.random.shuffle(X_train)
# np.random.seed(12)
# np.random.shuffle(y_train)
# np.random.seed(12)
# np.random.shuffle(X_test)
# np.random.seed(12)
# np.random.shuffle(y_test)
print("打乱训练集各个标签的出现次数",Counter(y_train))
print("打乱测试集各个标签的出现次数",Counter(y_test))
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#============================lightgbm=========================
from lightgbm.sklearn import LGBMClassifier
## 定义 LightGBM 模型 
lgbm= LGBMClassifier(class_weight="balanced")
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
joblib.dump(lgbm,'/home/u2208283040/tzx/LSTM/LightGBM/LightGBM.dat')
print("模型保存成功")
np.save(file="/home/u2208283040/tzx/LSTM/LightGBM/y_pred.npy",arr=y_pred)
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
plt.savefig('/home/u2208283040/tzx/LSTM/LightGBM/LightGBM.eps')
plt.savefig('/home/u2208283040/tzx/LSTM/LightGBM/LightGBM.png')
plt.show()
print("混淆矩阵保存成功")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("==================================ligbm_seq======================================")
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
print("==================================ligbm_seq======================================")
# #============================lightgbm=========================


#============================rm=========================
from sklearn.ensemble import RandomForestClassifier

rm = RandomForestClassifier(n_estimators=100, random_state=42,class_weight="balanced")
rm.fit(X_train, y_train)
y_pred = rm.predict(X_test)
joblib.dump(rm,'/home/u2208283040/tzx/LSTM/rm/rm.dat')
print("模型保存成功")
np.save(file="/home/u2208283040/tzx/LSTM/rm/y_pred.npy",arr=y_pred)
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
plt.savefig('/home/u2208283040/tzx/LSTM/rm/rm.eps')
plt.savefig('/home/u2208283040/tzx/LSTM/rm/rm.png')
plt.show()
print("混淆矩阵保存成功")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("==================================rm_seq======================================")
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
print("==================================rm_seq======================================")
# #============================rm=========================

#============================DecisionTree=========================
from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(random_state=42,class_weight="balanced")
DecisionTree.fit(X_train, y_train)
y_pred = DecisionTree.predict(X_test)
joblib.dump(DecisionTree,'/home/u2208283040/tzx/LSTM/DecisionTree/DecisionTree.dat')
print("模型保存成功")
np.save(file="/home/u2208283040/tzx/LSTM/DecisionTree/y_pred.npy",arr=y_pred)
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
plt.savefig('/home/u2208283040/tzx/LSTM/DecisionTree/DecisionTree.eps')
plt.savefig('/home/u2208283040/tzx/LSTM/DecisionTree/DecisionTree.png')
plt.show()
print("混淆矩阵保存成功")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("==================================DecisionTree_seq======================================")
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
print("==================================DecisionTree_seq======================================")
# #============================DecisionTree=========================

#============================xgboost=========================
import xgboost as xgb
xgboost = xgb.XGBClassifier()
xgboost.fit(X_train, y_train)
y_pred = xgboost.predict(X_test)
joblib.dump(xgboost,'/home/u2208283040/tzx/LSTM/xgboost/xgboost.dat')
print("模型保存成功")
np.save(file="/home/u2208283040/tzx/LSTM/xgboost/y_pred.npy",arr=y_pred)
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
plt.savefig('/home/u2208283040/tzx/LSTM/xgboost/xgboost.eps')
plt.savefig('/home/u2208283040/tzx/LSTM/xgboost/xgboost.png')
plt.show()
print("混淆矩阵保存成功")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("==================================xgboost_seq======================================")
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
print("==================================xgboost_seq======================================")
# #============================xgboost=========================
