import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 假设你有三个模型的预测概率和真实标签
y_true = np.load(file="/home/u2208283040/tzx/LSTM/y_ture_Kmer.npy")
y_pred_model1 = np.load(file="/home/u2208283040/tzx/LSTM/LightGBM/y_pred_Kmer.npy")
y_pred_model2 = np.load(file="/home/u2208283040/tzx/LSTM/rm/y_pred_Kmer.npy")
y_pred_model3 = np.load(file="/home/u2208283040/tzx/LSTM/DecisionTree/y_pred_Kmer.npy")
# y_pred_model4 = np.load(file="/home/u2208283040/tzx/LSTM/xgboost/y_pred_Kmer.npy")
print("加载")
# y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
# y_pred_model1 = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.6, 0.5, 0.9, 0.3, 0.2])
# y_pred_model2 = np.array([0.2, 0.3, 0.8, 0.7, 0.6, 0.5, 0.4, 0.9, 0.1, 0.3])
# y_pred_model3 = np.array([0.3, 0.6, 0.4, 0.9, 0.8, 0.5, 0.2, 0.7, 0.1, 0.4])
# y_pred_model4 = np.array([0.4, 0.5, 0.3, 0.5, 0.5, 0.6, 0.2, 0.4, 0.6, 0.4])
# 计算每个模型的ROC曲线
print(y_true.shape)
print(y_pred_model1.shape)
print(y_pred_model2.shape)
print(y_pred_model3.shape)
# print(y_pred_model4.shape)
fpr1, tpr1, _ = roc_curve(y_true, y_pred_model1)
fpr2, tpr2, _ = roc_curve(y_true, y_pred_model2)
fpr3, tpr3, _ = roc_curve(y_true, y_pred_model3)
# fpr4, tpr4, _ = roc_curve(y_true, y_pred_model4)
print("计算")
# 计算每个模型的AUC值
roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)
# roc_auc4 = auc(fpr4, tpr4)
print("绘制")
# 绘制ROC曲线
plt.figure()
plt.plot(fpr1, tpr1, color='darkorange', lw=1.3, label='LightGBM (AUC = %0.4f)' % roc_auc1)
plt.plot(fpr2, tpr2, color='cornflowerblue', lw=1.3, label='Random Forest (AUC = %0.4f)' % roc_auc2)
plt.plot(fpr3, tpr3, color='darkseagreen', lw=1.3, label='Decision Tree (AUC = %0.4f)' % roc_auc3)
# plt.plot(fpr4, tpr4, color='lightpink', lw=1.3, label='XGBoost (AUC = %0.4f)' % roc_auc4)
# 绘制对角线
plt.plot([0, 1], [0, 1], color='firebrick', lw=1.3, linestyle='--')

plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Different Models')
plt.legend(loc="lower right")
plt.savefig('/home/u2208283040/tzx/LSTM/roc.eps')
plt.savefig('/home/u2208283040/tzx/LSTM/roc.png')
plt.savefig('/home/u2208283040/tzx/LSTM/roc.pdf')
plt.show()
